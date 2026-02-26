import argparse
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration
import json
from tqdm import tqdm
import pandas as pd

def add_ranks(df, sort_by='score', ascending=False, groupby='query_id'):
    df = df.sort_values(by=[groupby, sort_by], ascending=[True, ascending])
    df['rank'] = df.groupby(groupby).cumcount() + 1
    return df


class MonoT5:
    def __init__(self, 
                 tok_model='t5-base',
                 model='castorini/monot5-base-msmarco',
                 batch_size=4,
                 text_field='text',
                 verbose=True):

        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return f"MonoT5({self.model_name})"

    def score_query(self, query: str, texts: list[str]) -> list[float]:
        """Score a single query against a list of passages."""
        scores = []
        prob = []

        it = range(0, len(texts), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus(
            ["Relevant:" for _ in range(self.batch_size)],
            return_tensors="pt",
            padding="longest",
        )
        max_vlen = self.model.config.n_positions - prompts["input_ids"].shape[1]

        if self.verbose:
            it = tqdm(it, desc="monoT5", unit="batches")

        for start_idx in it:
            rng = slice(start_idx, start_idx + self.batch_size)
            enc = self.tokenizer.batch_encode_plus(
                [
                    f"Query: {query} Document: {d}"
                    for d in texts[rng]
                ],
                return_tensors="pt",
                padding="longest",
            )

            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1]
                enc_value = enc_value[:, :max_vlen]
                enc[key] = torch.cat(
                    [enc_value, prompts[key][: enc_value.shape[0]]], dim=1
                )

            enc["decoder_input_ids"] = torch.full(
                (len(texts[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                result = self.model(**enc).logits

            result = result[:, 0, (self.REL, self.NREL)]
            log_probs = F.log_softmax(result, dim=1)

            scores_batch = log_probs[:, 0].cpu().detach().tolist()
            probs_batch = torch.exp(log_probs[:, 0]).cpu().detach().tolist()

            scores.extend(scores_batch)
            prob.extend(probs_batch)

        return prob

    def transform(self, run: pd.DataFrame):
        scores = [] # log prob
        prob = []

        queries, texts = run['query'], run[self.text_field]

        it = range(0, len(queries), self.batch_size)

        prompts = self.tokenizer.batch_encode_plus(['Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')

        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]

        if self.verbose:
            it = tqdm(it, desc='monoT5', unit='batches')

        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')

            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
                enc_value = enc_value[:, :max_vlen] # truncate any tokens that will not fit once the prompt is added

                enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end

            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                result = self.model(**enc).logits # result.shape: (batch_size, seq_len, vocab_size)

            result = result[:, 0, (self.REL, self.NREL)]

            log_probs = F.log_softmax(result, dim=1)

            scores_batch = log_probs[:, 0].cpu().detach().tolist()
            probs_batch = torch.exp(log_probs[:, 0]).cpu().detach().tolist()

            scores.extend(scores_batch)
            prob.extend(probs_batch)

            #scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()

        run = run.drop(columns=['score', 'rank'], errors='ignore')
        run = run.assign(score=scores, prob=prob)

        run = add_ranks(run)
        return run

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name", type=str)

    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--rerank_output_path", type=str)
    parser.add_argument("--qrels_output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    monoT5 = MonoT5(model=args.model_name_or_path, tok_model=args.tokenizer_name, batch_size=args.batch_size)


    print(f"Reading {args.dataset_path}")
    lines = []
    with open(args.dataset_path, 'r', encoding='utf-8') as r:
        for line in r:
            data = json.loads(line.strip())

            lines.append({
                'query_id': data['query_id'],
                'docid': data['docid'],
                'query': data['query'],
                'text': data['text'],
            })

    df = pd.DataFrame(lines)
    df = monoT5.transform(df)

    print(f"Writing reranked results to {args.rerank_output_path}")
    with open(args.rerank_output_path, 'w', encoding='utf-8') as w:
        for _, row in df.iterrows():
            #line = f"{row['query_id']} Q0 {row['docid']} {row['rank']} {row['score']} monot5"
            line = f"{row['query_id']} Q0 {row['docid']} {row['rank']} {row['prob']} monot5"
            w.write(line + "\n")

    print(f"Writing qrels results to {args.rerank_output_path}")
    with open(args.qrels_output_path, 'w', encoding='utf-8') as w:
        for _, row in df.iterrows():

            if float(row['prob'])>=0.5:
                label = 1
            else:
                label = 0
            line = f"{row['query_id']} 0 {row['docid']} {label}"
            w.write(line + "\n")    
"""