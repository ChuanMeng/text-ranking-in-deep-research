"""
BM25 searcher implementation using Pyserini.
"""

import json
import logging
from typing import Any, Dict, Optional

from pyserini.search.lucene import LuceneSearcher

from .base import BaseSearcher

from datasets import load_dataset

from ..query_rewriters.q2q import QueryToQuestion

from ..rerankers.rank1 import Rank1
from ..rerankers.monot5 import MonoT5
from ..rerankers.rankllama import RankLlama

import torch

logger = logging.getLogger(__name__)

class BM25Searcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):

        parser.add_argument(
            "--k1",
            type=float,
            default=0.9,
            help="k1 parameter for BM25",
        )
        parser.add_argument(
            "--b",
            type=float,
            default=0.4,
            help="b parameter for BM25",
        )

        parser.add_argument(
            "--rm3",
            action="store_true",
            help="RM3 query expansion",
        )

        parser.add_argument(
            "--index-path",
            required=True,
            help="Name or path of Lucene index (e.g. msmarco-v1-passage).",
        )
        parser.add_argument(
            "--get-document-dataset-name",
            default=None,
            help="Document dataset name for get_document function",
        )

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
            choices=["rank1", "monot5", "rankllama"],
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
            raise ValueError("index_path is required for BM25 searcher")

        self.args = args
        self.searcher = None

        logger.info(f"Initializing BM25 searcher with index: {args.index_path}")

        try:
            self.searcher = LuceneSearcher(args.index_path)
            self.searcher.set_bm25(k1=args.k1, b=args.b)
            logger.info(f"BM25 parameters set successfully with k1={args.k1} and b={args.b}")

            if args.rm3:
                self.searcher.set_rm3()
                logger.info("RM3 query expansion initialized successfully")

        except Exception as exc:
            raise ValueError(
                f"Index '{args.index_path}' is not a valid local Lucene index path."
            ) from exc

        logger.info("BM25 searcher initialized successfully")

        if args.get_document_dataset_name:
            logger.info(f"Loading dataset: {self.args.get_document_dataset_name}")
            ds = load_dataset(self.args.get_document_dataset_name, split="train")
            self.get_document_docid_to_text = {row["docid"]: row["text"] for row in ds}

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

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        if self.query_rewriter:
            if self.args.rewrite_with_context:
                reasoning_text = getattr(self, "last_reasoning")
                query = self.query_rewriter.rewrite_with_context(query, reasoning_text)
            else:
                query = self.query_rewriter.rewrite(query)

        if self.reranker:
            hits = self.searcher.search(query, self.args.reranking_depth)
        else:
            hits = self.searcher.search(query, k)

        retrieved_results = []
        for hit in hits:
            raw = json.loads(hit.lucene_document.get("raw"))
            retrieved_results.append(
                {"docid": hit.docid, "score": hit.score, "text": raw["contents"]}
            )
        
        if not self.reranker:
            return retrieved_results

        # re-ranking
        assert len(retrieved_results)<=self.args.reranking_depth
        candidates = retrieved_results

        texts = [item["text"] for item in candidates]
        scores = self.reranker.score_query(query, texts)

        reranked_pairs = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
        reranked = [
                {"docid": item["docid"], "score": float(score), "text": item["text"]}
                for item, score in reranked_pairs]

        return reranked[:self.args.k]
        #return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        doc = self.searcher.doc(docid)
        
        if doc is None:
            raise RuntimeError(f"Document {docid} is Not Found")

        return {
            "docid": docid,
            "text": json.loads(doc.raw())["contents"],
        }

    @property
    def search_type(self) -> str:
        return "BM25"
