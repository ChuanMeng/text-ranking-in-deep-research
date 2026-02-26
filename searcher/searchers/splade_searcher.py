"""
SPLADE searcher implementation using Pyserini.
"""

import json
import logging
import numpy as np
from typing import Any, Dict, Optional

from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer
from tevatron.retriever.modeling import EncoderOutput, SpladeModel
from datasets import load_dataset
from pathlib import Path

from .base import BaseSearcher

from ..query_rewriters.q2q import QueryToQuestion

from ..rerankers.rank1 import Rank1
from ..rerankers.monot5 import MonoT5
from ..rerankers.rankllama import RankLlama

import torch
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

class SpladeSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--index-path",
            required=True,
            help="Name or path of Lucene index (e.g. msmarco-v1-passage).",
        )
        parser.add_argument("--model-name", required=True, help="SPLADE model, e.g. naver/splade-v3")
        parser.add_argument("--tokenizer-name", required=True, help="Tokenizer, e.g. bert-base-uncased")

        parser.add_argument(
            "--max-length",
            type=int,
            default=512,
            help="Maximum sequence length for SPLADE search (default: 512)",
        )
        parser.add_argument(
            "--dataset-name",
            help="Dataset name for document retrieval in Splade search (default: Tevatron/browsecomp-plus-corpus)",
        )

        parser.add_argument(
            "--get-document-dataset-name",
            default=None,
            help="Document dataset name for get_document function",
        )

        parser.add_argument("--splade_query", action="store_true", help="Use SPLADED query encoded by Tevatron for retrieval")

        # Query Rewriter
        parser.add_argument(
            "--query-rewriter-type",
            default=None,
            choices=["q2q"],
            help="Query rewriter to apply before retrieval",
        )
        parser.add_argument(
            "--rewrite_with_context",
            action="store_true",
            help="Rewrite query with context",
        )
        parser.add_argument(
            "--q2q-model",
            default="openai/gpt-oss-20b",
            help="Backbone model for Q2Q rewriting",
        )
        parser.add_argument(
            "--q2q-model-url",
            default="http://localhost:8000/v1",
            help="Model URL for Q2Q rewriting",
        )
        parser.add_argument(
            "--q2q-max-new-tokens",
            type=int,
            default=64,
            help="Max new tokens for Q2Q generation",
        )
        
        # Reranker
        parser.add_argument(
            "--reranker-type",
            default=None,
            choices=["monot5", "rankllama","rank1"],
            help="Reranker to apply after retrieval",
        )
        parser.add_argument(
            "--reranking-depth",
            type=int,
            default=50,
            help="Number of documents to return after reranking",
        )
        parser.add_argument(
            "--rank1-model",
            default=None,
            help="Rank1 model name/path for reranking (e.g. jhu-clsp/rank1-7b)",
        )
        parser.add_argument(
            "--rank1-model-url",
            default=None,
            help="Model URL for Rank1 reranking",
        )
        parser.add_argument(
            "--rank1-batch-size",
            type=int,
            default=32,
            help="Batch size for Rank1 reranking",
        )
        parser.add_argument(
            "--rank1-context-size",
            type=int,
            default=450,
            help="Context size for Rank1 reranking",
        )
        parser.add_argument(
            "--rank1-max-output-tokens",
            type=int,
            default=2000,
            help="Max output tokens for Rank1 reranking",
        )
        parser.add_argument(
            "--monot5-model",
            default="castorini/monot5-base-msmarco",
            help="MonoT5 model name/path",
        )
        parser.add_argument(
            "--monot5-tokenizer",
            default="t5-base",
            help="MonoT5 tokenizer name/path",
        )
        parser.add_argument(
            "--monot5-batch-size",
            type=int,
            default=4,
            help="Batch size for MonoT5 reranking",
        )
        parser.add_argument(
            "--rankllama-model",
            default=None,
            help="RankLlama model name/path for reranking",
        )
        parser.add_argument(
            "--rankllama-tokenizer",
            default=None,
            help="RankLlama tokenizer name/path",
        )
        parser.add_argument(
            "--rankllama-lora",
            default=None,
            help="RankLlama model name/path for reranking",
        )
        parser.add_argument(
            "--rankllama-batch-size",
            type=int,
            default=8,
            help="Batch size for RankLlama reranking",
        )
        parser.add_argument(
            "--rankllama-max-len",
            type=int,
            default=512,
            help="Max sequence length for RankLlama reranking",
        )
        parser.add_argument(
            "--rankllama-query-prefix",
            default="query: ",
            help="Query prefix/instruction for RankLlama reranking",
        )
        parser.add_argument(
            "--rankllama-passage-prefix",
            default="document: ",
            help="Passage prefix/instruction for RankLlama reranking",
        )
        parser.add_argument(
            "--rankllama-append-eos",
            action="store_true",
            help="Append EOS token in RankLlama reranking",
        )
        parser.add_argument(
            "--rankllama-fp16",
            action="store_true",
            help="Use fp16 autocast in RankLlama reranking",
        )



    def __init__(self, args):
        if not args.index_path:
            raise ValueError("index_path is required for Splade searcher")
        # Initialize the searcher with the arguments
        self.args = args

        logger.info(f"Initializing Splade searcher with index: {args.index_path}")

        self.searcher = LuceneImpactSearcher(args.index_path, None, 0)
        analyzer = JWhiteSpaceAnalyzer()
        self.searcher.set_analyzer(analyzer)

        logger.info("Splade searcher initialized successfully")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading SPLADE encoder: model={args.model_name}, tokenizer={args.tokenizer_name}, device={self.device}"
        )
        
        # IMPORTANT: tokenizer should be bert-base-uncased; model should be naver/splade-v3
        config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
        self.model = SpladeModel.load(args.model_name, config=config)

        self.model = self.model.to(self.device)
        self.model.eval()

        vocab_dict = self.tokenizer.get_vocab()
        self.vocab_dict = {v: k for k, v in vocab_dict.items()}

        self._load_dataset()

        if args.get_document_dataset_name:
            logger.info(f"Loading dataset: {self.args.get_document_dataset_name}")
            ds = load_dataset(self.args.get_document_dataset_name, split="train")
            self.get_document_docid_to_text = {row["docid"]: row["text"] for row in ds}

        logger.info("SPLADE searcher initialized successfully")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.query_rewriter = None
        if args.query_rewriter_type == "q2q":
            logger.info("Loading Q2Q rewriter: %s", args.q2q_model)
            self.query_rewriter = QueryToQuestion(
                model_name=args.q2q_model,
                model_url=args.q2q_model_url,
                max_new_tokens=args.q2q_max_new_tokens,
            )
            logger.info("Q2Q rewriter initialized successfully")

        self.reranker = None
        if args.reranker_type == "monot5":
            logger.info("Loading MonoT5 reranker: %s", args.monot5_model)
            self.reranker = MonoT5(
                model=args.monot5_model,
                tok_model=args.monot5_tokenizer,
                batch_size=args.monot5_batch_size,
                verbose=False,
            )
            logger.info("MonoT5 reranker initialized successfully")
        elif args.reranker_type == "rankllama":
            if not args.rankllama_model:
                raise ValueError("--rankllama-model is required for RankLlama reranking")
            logger.info("Loading RankLlama reranker: %s", args.rankllama_model)
            self.reranker = RankLlama(
                model_name_or_path=args.rankllama_model,
                tokenizer_name=args.rankllama_tokenizer,
                lora_name_or_path=args.rankllama_lora,
                batch_size=args.rankllama_batch_size,
                rerank_max_len=args.rankllama_max_len,
                query_prefix=args.rankllama_query_prefix,
                passage_prefix=args.rankllama_passage_prefix,
                append_eos_token=args.rankllama_append_eos,
                fp16=args.rankllama_fp16,
                device=self.device,
            )
            logger.info("RankLlama reranker initialized successfully")
        elif args.reranker_type == "rank1":
            if not args.rank1_model:
                raise ValueError("--rank1-model is required for Rank1 reranking")
            logger.info("Loading Rank1 reranker: %s", args.rank1_model)
            self.reranker = Rank1(
                model=args.rank1_model,
                batch_size=args.rank1_batch_size,
                context_size=args.rank1_context_size,
                max_output_tokens=args.rank1_max_output_tokens,
                device=self.device,
                api_url=args.rank1_model_url,
            )
            logger.info("Rank1 reranker initialized successfully")

    def _load_dataset(self) -> None:
        
        if Path(self.args.dataset_name).is_file():
            logger.info(f"Loading dataset: {self.args.dataset_name}")
            self.docid_to_text = dict()
            with open(self.args.dataset_name, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.docid_to_text[data["docid"]] = data["title"]+" "+data["text"] if data["title"] else data["text"]
            logger.info(f"Loaded {len(self.docid_to_text)} passages from dataset")
        else:
            logger.info(f"Loading dataset: {self.args.dataset_name}")
            ds = load_dataset(self.args.dataset_name, split="train")
            self.docid_to_text = {row["docid"]: row["text"] for row in ds}
            logger.info(f"Loaded {len(self.docid_to_text)} passages from dataset")

    def _encode_query_to_splade_string(self, query: str) -> str:
        # === tokenizer ===
        batch = self.tokenizer(
            query,
            padding=False, 
            truncation=True,
            max_length=self.args.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        with torch.cuda.amp.autocast(): # if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # === SPLADE forward ===
                # encode_is_query=True => use q_reps
                model_output: EncoderOutput = self.model(query=batch)
                reps = model_output.q_reps.cpu().detach().numpy()   # shape: (1, vocab)

                rep = reps[0] # (vocab,)
                # === EXACT SAME LOGIC AS encode_splade.py ===
                idx = np.nonzero(rep)
                data = rep[idx] # len(idx[0])
                data = np.rint(data * 100).astype(int)

                dict_splade = dict()
                for id_token, value_token in zip(idx[0], data):
                    if value_token > 0:
                        real_token = self.vocab_dict[id_token]
                        dict_splade[real_token] = int(value_token)

                if len(dict_splade.keys()) == 0:
                    # same fallback token as original
                    #print("empty input =>", query)
                    dict_splade[self.vocab_dict[998]] = 1

                # === build Lucene pretokenized query string ===
                string_splade = " ".join(
                    [" ".join([str(real_token)] * freq)
                     for real_token, freq in dict_splade.items()]
                )

                return string_splade

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:

        #print("raw query:", query)
        if self.query_rewriter:
            if self.args.splade_query:
                raise ValueError("SPLADE query should not be rewritten")

            if self.args.rewrite_with_context:
                reasoning_text = getattr(self, "last_reasoning")
                #print("reasoning text:", reasoning_text, flush=True)
                query = self.query_rewriter.rewrite_with_context(query, reasoning_text)
            else:
                query = self.query_rewriter.rewrite(query)
        
        #print("rewritten query:", query)
        
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")
        
        if not self.args.splade_query:
            splade_query = self._encode_query_to_splade_string(query)
            if self.reranker:
                hits = self.searcher.search(splade_query, self.args.reranking_depth)
            else:
                hits = self.searcher.search(splade_query, k)
        else:
            splade_query = query
            if self.reranker:
                hits = self.searcher.search(query, self.args.reranking_depth)
            else:
                hits = self.searcher.search(query, k)
        
        #hits = self.searcher.search(splade_query, k)
        retrieved_results = []
        for hit in hits:
            passage_text = self.docid_to_text[hit.docid]
            retrieved_results.append(
                {"docid": hit.docid, "score": hit.score, "text": passage_text}
            )
        
        if not self.reranker:
            return retrieved_results

        # re-ranking
        assert len(retrieved_results)==self.args.reranking_depth
        candidates = retrieved_results

        texts = [item["text"] for item in candidates]
        scores = self.reranker.score_query(query, texts)

        reranked_pairs = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
        reranked = [
                {"docid": item["docid"], "score": float(score), "text": item["text"]}
                for item, score in reranked_pairs]

        return reranked[:k]

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.docid_to_text:
            raise RuntimeError("Dataset not loaded")

        text = self.docid_to_text.get(docid)
        if text is None:
            raise RuntimeError(f"Document {docid} is Not Found")

        return {
            "docid": docid,
            "text": text,
        }
    
    
    @property
    def search_type(self) -> str:
        return "SPLADE"