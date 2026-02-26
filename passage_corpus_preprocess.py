import argparse
import json
import logging
from pathlib import Path

import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_browsecomp_plus_passage(
    input_dir: Path,
    output_dir: Path,
):

    doc_count = set()

    pyserini_dir = output_dir / "browsecomp-plus-passage-pyserini"

    output_dir.mkdir(parents=True, exist_ok=True)
    pyserini_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Preprocessing BrowseComp-Plus passage corpus...")

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    nlp = spacy.load(
        "en_core_web_sm",
        exclude=[
            "parser",
            "tagger",
            "ner",
            "attribute_ruler",
            "lemmatizer",
            "tok2vec",
        ],
    )

    nlp.enable_pipe("senter")
    spacy_max_len = 250
    bert_max_len = 512
    qwen_max_len = qwen_tokenizer.model_max_length
    if qwen_max_len is None or qwen_max_len > 1_000_000:
        assert qwen_max_len == 131072
    
    count = 0
    whitespace_total = 0
    bert_total = 0
    qwen_total = 0
    spacy_total = 0

    bert_over = 0
    qwen_over = 0
    spacy_over = 0


    pyserini_path = pyserini_dir / "corpus.jsonl"
    tevatron_path = output_dir / "browsecomp_plus-passage-tevatron.jsonl"

    with pyserini_path.open("w", encoding="utf-8") as w1, tevatron_path.open("w", encoding="utf-8") as w2:
        for input_path in sorted(input_dir.glob("BrowseComp-Plus_*.jsonl")):

            logging.info(f"Processing {input_path}")

            with input_path.open("r", encoding="utf-8") as r:
                total_lines = sum(1 for _ in r)
                
            with input_path.open("r", encoding="utf-8") as r:
                for line in tqdm(r, total=total_lines, desc=input_path.name):
                    line = line.strip()
                    record = json.loads(line)
                    
                    #record = None
                    #try:
                    #    record = json.loads(line)
                    #except Exception:
                    #    print("ERROR: Invalid JSON:\n", line)
                    #    raise Exception

                    doc_id = record["id"]
                    passages = record["contents"]
                    title = record["title"]

                    doc_count.add(doc_id)

                    for passage in passages:
                        idx = passage["id"]
                        body = passage["body"]

                        passage_id = f"{doc_id}_{idx}"
                        w1.write(
                            json.dumps({"id": passage_id, "contents": f"{title} {body}" if title else body})
                            + "\n"
                        )
                        w2.write(
                            json.dumps(
                                {"docid": passage_id, "title": title, "text": body}
                            )
                            + "\n"
                        )
                        whitespace_len = len(body.split())
                        bert_len = len(bert_tokenizer.tokenize(body))
                        qwen_len = len(qwen_tokenizer.tokenize(body))

                        spacy_len = len(nlp(body))

                        whitespace_total += whitespace_len
                        bert_total += bert_len
                        qwen_total += qwen_len
                        spacy_total += spacy_len
                        
                        if bert_len > bert_max_len:
                            bert_over += 1
                            print(
                                f"WARNING: bert>{bert_max_len} passage={passage_id} bert_len={bert_len} spacy_len={spacy_len} ws={whitespace_len}"
                            )
                        if qwen_len > qwen_max_len:
                            qwen_over += 1
                            print(
                                f"WARNING: qwen>{qwen_max_len} passage={passage_id} qwen_len={qwen_len} spacy_len={spacy_len} ws={whitespace_len}"
                            )
                        if spacy_len > spacy_max_len:
                            print(
                                f"WARNING: spacy>{spacy_max_len} passage={passage_id} bert_len={bert_len} spacy_len={spacy_len} ws={whitespace_len} \n{body}\n\n"
                            )
                            spacy_over += 1
                        count += 1

    logging.info(f"# processed documents: {len(doc_count)}")
    logging.info(f"# passages: {count}")
    
    if count:
        logging.info(f"# avg passage length (whitespace): {whitespace_total / count:.2f}")
        logging.info(f"# avg passage length (bert): {bert_total / count:.2f}")
        logging.info(f"# avg passage length (qwen3): {qwen_total / count:.2f}")
        logging.info(f"# avg passage length (spacy): {spacy_total / count:.2f}")
        
    logging.info(f"# passages over bert max ({bert_max_len}): {bert_over}")
    logging.info(f"# passages over qwen max ({qwen_max_len}): {qwen_over}")
    logging.info(f"# passages over spacy max ({spacy_max_len}): {spacy_over}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess BrowseComp-Plus passage corpus.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("./data/jsonlines"),
        help="Directory with BrowseComp-Plus JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./data/browsecomp-plus-passage"),
        help="Output directory for processed passage corpus.",
    )
    args = parser.parse_args()

    preprocess_browsecomp_plus_passage(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )