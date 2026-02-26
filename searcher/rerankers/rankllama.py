from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer
from contextlib import nullcontext

from tevatron.reranker.arguments import DataArguments
from tevatron.reranker.collator import RerankerInferenceCollator
from tevatron.reranker.dataset import format_pair
from tevatron.reranker.modeling import RerankerModel


@dataclass
class RankLlama:
    """RankLlama reranker using Tevatron inference components."""

    model_name_or_path: str
    tokenizer_name: str | None = None
    lora_name_or_path: str | None = None
    batch_size: int = 8
    rerank_max_len: int = 512
    query_prefix: str = "query: "
    passage_prefix: str = "document: " 
    append_eos_token: bool = False
    pad_to_multiple_of: int = 16
    fp16: bool = True
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name or self.model_name_or_path
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "right"

        self.model = RerankerModel.load(
            self.model_name_or_path,
            lora_name_or_path=self.lora_name_or_path,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.data_args = DataArguments(
            rerank_max_len=self.rerank_max_len,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
            append_eos_token=self.append_eos_token,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        self.collator = RerankerInferenceCollator(
            data_args=self.data_args, tokenizer=self.tokenizer
        )

    def score_query(self, query: str, texts: List[str]) -> List[float]:
        """Score a single query against a list of passages."""
        if not texts:
            return []

        features = [
            ("q0", str(i), format_pair(query, text, "", self.query_prefix, self.passage_prefix))
            for i, text in enumerate(texts)
        ]

        scores: List[float] = []
        #use_autocast = self.fp16 and self.device.startswith("cuda")
        #autocast_ctx = torch.cuda.amp.autocast() if use_autocast else nullcontext()

        for start in range(0, len(features), self.batch_size):
            batch_features = features[start : start + self.batch_size]
            _, _, batch = self.collator(batch_features)

            with torch.cuda.amp.autocast() if self.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                
                    model_output = self.model(batch)
                    batch_scores = model_output.scores.cpu().detach().float().numpy()

            for i in range(len(batch_scores)):
                scores.append(float(batch_scores[i][0]))

        return scores
