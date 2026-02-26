from pathlib import Path
import argparse
import json
import math
from pylate import indexes, models, retrieve
from datasets import load_dataset
from tqdm import tqdm
import torch

def main(args):
    model = models.ColBERT(
        args.model_name,
        document_length=args.document_length,
        query_length=args.query_length,
    )
    model.max_seq_length = args.max_seq_length

    def build_index_from_records(records, index_name: str) -> None:
        index = indexes.PLAID(
            index_folder=args.index_folder,
            index_name=index_name,
            override=True,
        )
        if args.chunk_size > 0:
            buf_ids, buf_texts = [], []
            for record in tqdm(
                records, desc=f"Indexing {index_name} (chunk_size={args.chunk_size})"
            ):
                buf_ids.append(record["docid"])
                buf_texts.append(record["text"])

                if len(buf_texts) >= args.chunk_size:
                    emb = model.encode(
                        buf_texts,
                        batch_size=args.batch_size,
                        is_query=False,
                        show_progress_bar=False,
                    )
                    index.add_documents(
                        documents_ids=buf_ids, documents_embeddings=emb
                    )

                    del emb 
                    torch.cuda.empty_cache()

                    buf_ids, buf_texts = [], []

            if buf_texts:
                emb = model.encode(
                    buf_texts,
                    batch_size=args.batch_size,
                    is_query=False,
                    show_progress_bar=False,
                )
                index.add_documents(documents_ids=buf_ids, documents_embeddings=emb)

                del emb 
                torch.cuda.empty_cache()
        else:
            documents_ids = []
            documents = []

            for record in tqdm(records, desc=f"Loading documents for {index_name}"):
                documents_ids.append(record["docid"])
                documents.append(record["text"])
            
            # for testing
            #documents_ids = documents_ids[:1000]
            #documents = documents[:1000]

            print(f"Loaded {len(documents)} documents for {index_name}")
            documents_embeddings = model.encode(
                documents,
                batch_size=args.batch_size,
                is_query=False,
                show_progress_bar=True, # too slow if True
            )
            index.add_documents(
                documents_ids=documents_ids,
                documents_embeddings=documents_embeddings,
            )

            del documents_embeddings
            torch.cuda.empty_cache()

    # build index from records
    if args.build_index:
        corpus_path = Path(args.corpus_path)
        if corpus_path.is_file():
            dataset = []
            with corpus_path.open("r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    title = data["title"]
                    text = data["text"]
                    full_text = f"{title} {text}".strip() if title else text
                    dataset.append({"docid": data["docid"], "text": full_text})
        else:
            dataset = load_dataset(args.corpus_path, split="train")

        shard_index_names = []
        if args.shard_count > 1:
            total_records = len(dataset)
            shard_size = max(1, math.ceil(total_records / args.shard_count))
            shard_records = []
            shard_idx = 0
            for record in dataset:
                shard_records.append(record)
                if len(shard_records) >= shard_size:
                    shard_name = f"index_shard_{shard_idx}"
                    print(
                        f"Starting shard {shard_idx} with {len(shard_records)} documents"
                    )
                    build_index_from_records(shard_records, shard_name)
                    print(f"Finished shard {shard_idx}")
                    shard_index_names.append(shard_name)
                    shard_idx += 1
                    shard_records = []
            if shard_records:
                shard_name = f"index_shard_{shard_idx}"
                print(
                    f"Starting shard {shard_idx} with {len(shard_records)} documents"
                )
                build_index_from_records(shard_records, shard_name)
                print(f"Finished shard {shard_idx}")
                shard_index_names.append(shard_name)
        else:
            # no sharding
            build_index_from_records(dataset, "index")
            shard_index_names = ["index"]
    else:
        # load index only
        shard_index_names = []
        if args.shard_count > 1:
            index_folder = Path(args.index_folder)
            shard_index_names = sorted(
                p.name for p in index_folder.glob("index_shard_*")
            )
        if not shard_index_names:
            shard_index_names = ["index"]

        query_path = Path(args.query_path)
        queries_by_id = {}

        with query_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    qid, text = parts
                    queries_by_id[qid] = text
                    #break
                    
                    raise Exception(f"Invalid line: {line}")
                qid, text = parts
                queries_by_id[qid] = text
                #break
                
        queries_embeddings = model.encode(
            list(queries_by_id.values()),
            batch_size=32,
            is_query=True, # Encoding queries
            show_progress_bar=True,
        )
        """
        scores = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=args.k,
        )
        """

        scores = []
        num_q = len(queries_by_id)

        for start in tqdm(range(0, num_q, args.search_batch), desc="Searching"):
            end = min(start + args.search_batch, num_q)
            q_emb_batch = queries_embeddings[start:end]

            merged_hits = [[] for _ in range(len(q_emb_batch))]
            for index_name in shard_index_names:
                index = indexes.PLAID(
                    index_folder=args.index_folder,
                    index_name=index_name,
                    override=False,
                )
                retriever = retrieve.ColBERT(index=index)
                scores_batch = retriever.retrieve(
                    queries_embeddings=q_emb_batch,
                    k=args.k,
                )
                for idx, hits in enumerate(scores_batch):
                    merged_hits[idx].extend(hits)
                    
                del retriever
                del index
                torch.cuda.empty_cache()

            for hits in merged_hits:
                hits.sort(key=lambda x: x["score"], reverse=True)
                scores.append(hits[: args.k])


        run_path = Path(args.run_path)

        qids = list(queries_by_id.keys())
        run_tag = args.model_name

        with run_path.open("w", encoding="utf-8") as f:
            for qid, hits in zip(qids, scores):
                for rank, hit in enumerate(hits, start=1):
                    docid = hit["id"]
                    score = hit["score"]
                    f.write(f"{qid} Q0 {docid} {rank} {score} {run_tag}\n")

        print(f"Wrote run file to {run_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="colbert-ir/colbertv2.0")
    parser.add_argument("--corpus_path", type=str, default="Tevatron/msmarco-passage-corpus")
    parser.add_argument("--query_path", type=str, default="./topics-qrels/topics.msmarco-passage.dev-subset.txt")
    parser.add_argument("--index_folder", type=str, default="./indexes/index.colbertv2.0.msmarco-passage")
    parser.add_argument("--run_path", type=str, default="./runs/run.colbertv2.0.msmarco-passage.dev-subset.txt")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--shard_count",
        type=int,
        default=1,
        help="Number of shards to split the corpus into",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--document_length", type=int, default=300)
    parser.add_argument("--query_length", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=0)
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--search_batch", type=int, default=1)
    parser.add_argument("--k", type=int, default=1000)
    args = parser.parse_args()
    main(args)