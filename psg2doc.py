import argparse
import json
from collections import defaultdict
from pathlib import Path


def passage_to_docid(passage_id: str) -> str:
    # Passage ids are created as f"{doc_id}_{idx}"
    doc_id = passage_id[len("browsecomp_plus_") :].rsplit("_", 1)[0]
    return doc_id


def convert_run(input_path: Path, output_path: Path) -> None:
    doc_scores: dict[str, dict[str, float]] = defaultdict(dict)
    run_tags: dict[str, str] = {}
    passage_counts: dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid run line (expected >=6 fields): {line}")

            qid = parts[0]
            passage_id = parts[2]
            score = float(parts[4])
            tag = parts[5]

            docid = passage_to_docid(passage_id)
            passage_counts[qid] += 1
            prev = doc_scores[qid].get(docid)
            if prev is None or score > prev:
                doc_scores[qid][docid] = score
            if qid not in run_tags:
                run_tags[qid] = tag

    with output_path.open("w", encoding="utf-8") as out:
        for qid in sorted(doc_scores.keys()):
            tag = run_tags[qid]
            items = sorted(doc_scores[qid].items(), key=lambda kv: (-kv[1], kv[0]))
            for rank, (docid, score) in enumerate(items, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {score} {tag}\n")

    total_queries = len(doc_scores)
    total_docs = sum(len(docs) for docs in doc_scores.values())
    total_passages = sum(passage_counts.values())
    avg_passages = total_passages / total_queries if total_queries else 0.0
    avg_docs = total_docs / total_queries if total_queries else 0.0
    
    print(
        f"Converted run saved to {output_path}\n"
        f"Queries: {total_queries}\n"
        f"Input passages: {total_passages}\n"
        f"Avg passages/query: {avg_passages:.2f}\n"
        f"Documents: {total_docs}\n"
        f"Avg docs/query: {avg_docs:.2f}"
    )

def detect_zero_retrieved_docids(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    count = 0
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to read JSON file {json_path}: {exc}") from exc

        assert "retrieved_docids" in data, f"JSON file {json_path} does not have retrieved_docids"
        if len(data["retrieved_docids"]) == 1 and data["retrieved_docids"][0] == "":
            count += 1

    print(f"Found {count} JSON files with zero retrieved docids")


def convert_run_json_dir(input_dir: Path, output_dir: Path) -> None:
    """Convert per-query JSON run files in a directory from passage ids to doc ids.

    - Replaces `retrieved_docids` with document ids.
    - Preserves original passage ids under `retrieved_docids_passages`.
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    processed_count = 0
    #skipped_count = 0
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to read JSON file {json_path}: {exc}") from exc

        #if "retrieved_docids_passages" in data:
        #    skipped_count += 1
        #else:
        passage_ids = data["retrieved_docids"]
        #data["retrieved_docids_passages"] = list(passage_ids)
        docids_set = set()
        for pid in passage_ids:
            docids_set.add(passage_to_docid(pid))
        data["retrieved_docids"] = list(docids_set)
        processed_count += 1

        output_path = output_dir / json_path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Convert passage-id to doc-id: Processed {processed_count} JSON run files in {input_dir}")
    #print(f"Skipped {skipped_count} JSON run files in {input_dir} because they already have retrieved_docids_passages")
    print(f"Output written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert passage run to document run using maxP."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_passage_run",
        help="Path to passage run file (TREC format).",
    )
    group.add_argument(
        "--input_json_dir",
        help="Path to directory containing per-query JSON run files.",
    )
    parser.add_argument(
        "--output_json_dir",
        help="Path to output directory for converted JSON run files.",
    )
    parser.add_argument(
        "--output_document_run",
        help="Path to output document run file (TREC format).",
    )
    parser.add_argument(
        "--detect_zero_retrieved_docids",
        action="store_true",
        help="Detect JSON files with zero retrieved docids.",
    )

    args = parser.parse_args()

    if args.detect_zero_retrieved_docids:
        if not args.input_json_dir:
            raise ValueError("--input_json_dir is required with --detect_zero_retrieved_docids")
        detect_zero_retrieved_docids(Path(args.input_json_dir))
        return

    if args.input_json_dir:
        if not args.output_json_dir:
            raise ValueError("--output_json_dir is required with --input_json_dir")
        input_json_dir = Path(args.input_json_dir)
        output_json_dir = Path(args.output_json_dir)
        if input_json_dir.resolve() == output_json_dir.resolve():
            raise ValueError("--output_json_dir must be different from --input_json_dir")
        convert_run_json_dir(input_json_dir, output_json_dir)
        return

    if not args.output_document_run:
        raise ValueError("--output_document_run is required with --input_passage_run")

    input_passage_run = Path(args.input_passage_run)
    output_document_run = Path(args.output_document_run)
    convert_run(input_passage_run, output_document_run)


if __name__ == "__main__":
    main()