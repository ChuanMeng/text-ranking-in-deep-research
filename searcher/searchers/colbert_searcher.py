"""
ColBERT searcher implementation using Pylate.
"""

import json
import logging
from typing import Any, Dict, Optional

from datasets import load_dataset
from pathlib import Path
#from pylate import indexes, models, retrieve
import torch

from .base import BaseSearcher

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class ColbertSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument("--model-name", required=True, help="ColBERT model, e.g. colbert-ir/colbertv2.0")
        parser.add_argument("--index-path", required=True, help="Path to an existing PLAID index folder.")

        parser.add_argument(
            "--max-query-length",
            type=int,
            default=512,
            help="Maximum sequence length for Colbert search (default: 512)",
        )
        parser.add_argument(
            "--max-document-length",
            type=int,
            default=512,
            help="Maximum sequence length for Colbert search (default: 512)",
        )
        parser.add_argument(
            "--dataset-name",
            help="Dataset name for document retrieval (default: Tevatron/browsecomp-plus-corpus)",
        )
        

    def __init__(self, args):
        if not args.index_path:
            raise ValueError("index_path is required for Colbert searcher")
        # Initialize the searcher with the arguments
        self.args = args

        logger.info(
            "Initializing Colbert searcher with index: %s",
            args.index_path,
        )

        self.model = models.ColBERT(args.model_name, document_length=args.max_document_length, query_length=args.max_query_length)

        index_folder = Path(args.index_path)
        self.retriever = None


        shard_dirs = sorted(
            p.name for p in index_folder.glob("index_shard_*") if p.is_dir()
        )

        if shard_dirs:
            self.shard_index_names = shard_dirs
            logger.info(
                f"Detected {len(self.shard_index_names)} shard indexes: {', '.join(self.shard_index_names)}")
        else:
            index_dirs = [p.name for p in index_folder.iterdir() if p.is_dir()]
            if not index_dirs:
                raise ValueError(f"No index directories found in {index_folder}")
            if len(index_dirs) > 1:
                raise ValueError(
                    f"Multiple index directories found in {index_folder}: {index_dirs}. "
                    "Expected a single index folder or shard folders named index_shard_*."
                )
            auto_index_name = index_dirs[0]
            logger.info(
                "Detected single index in %s: %s",
                index_folder,
                auto_index_name,
            )
            index = indexes.PLAID(
                index_folder=args.index_path,
                index_name=auto_index_name,
                override=False,
            )
            self.retriever = retrieve.ColBERT(index=index)
            
            
        self._load_dataset()

        logger.info("Colbert searcher initialized successfully")


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

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        queries_embedding = self.model.encode(query,
            is_query=True, # Encoding queries
            #show_progress_bar=True,
        )

        if self.retriever is not None and not self.shard_index_names:
            hits = self.retriever.retrieve(
                queries_embeddings=queries_embedding,
                k=k,
            )
            assert len(hits) == 1
            hits = hits[0]
        else:
            merged_hits = []
            for index_name in self.shard_index_names:
                index = indexes.PLAID(
                    index_folder=self.args.index_path,
                    index_name=index_name,
                    override=False,
                )
                retriever = retrieve.ColBERT(index=index)
                hits = retriever.retrieve(
                    queries_embeddings=queries_embedding,
                    k=k,
                )
                assert len(hits) == 1
                merged_hits.extend(hits[0])
                del retriever
                del index
                torch.cuda.empty_cache()

            merged_hits.sort(key=lambda x: x["score"], reverse=True)
            hits = merged_hits[:k]
        
        results = []
        for hit in hits:
            passage_text = self.docid_to_text[hit["id"]]
            results.append(
                {"docid": hit["id"], "score": hit["score"], "text": passage_text}
            )

        return results

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
        return "COLBERT"