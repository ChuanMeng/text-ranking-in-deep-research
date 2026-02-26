import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count completed/incompleted runs in a directory of JSON files."
    )
    parser.add_argument(
        "--input_dir", required=True, help="Directory containing run JSON files"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    completed = 0
    incompleted = 0
    missing_status = 0
    missing_final_output = 0
    over_hundred_tool_calls = 0
    search_calls_missing_final_output = []
    load_errors = 0

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                run_data = json.load(f)
        except Exception:
            load_errors += 1
            continue

        #print(run_data["tool_call_counts"],run_data)
        if run_data["tool_call_counts"].get("search",0) > 100:
            over_hundred_tool_calls += 1


        result_seq = run_data["result"]
        #print(len(result_seq),run_data["tool_call_counts"]["search"])
        if result_seq[-1]["type"] != "output_text":
            missing_final_output += 1
            #search_calls_missing_final_output.append(run_data["tool_call_counts"]["search"])

            #assert run_data["status"] == "incompleted" or run_data["tool_call_counts"]["search"]>=100,f"status: {run_data['status']}, tool_call_counts: {run_data['tool_call_counts']},print(run_data)"
            #print(run_data)

        status = run_data.get("status")
        if status is None:
            missing_status += 1
            incompleted += 1
            continue

        if status == "completed":
            completed += 1
        else:
            incompleted += 1

    total = len(json_files)
    counted = completed + incompleted

    print(f"Input directory: {input_dir}")
    print(f"Total JSON files: {total}")
    completed_rate = (completed / total) if total else 0.0
    print(f"Completed: {completed}")
    print(f"Incompleted: {incompleted}")
    print(f"Missing status: {missing_status}")
    print(f"Load errors: {load_errors}")
    print(f"Missing final output: {missing_final_output}")
    #print(f"avg # Search calls for missing final output: {sum(search_calls_missing_final_output)/len(search_calls_missing_final_output)}")
    #print(f"min # Search calls for missing final output: {min(search_calls_missing_final_output)}")
    #print(f"max # Search calls for missing final output: {max(search_calls_missing_final_output)}")

    print(f"Over 100 tool calls: {over_hundred_tool_calls}")
    print(f"Completed rate (by total queries): {completed_rate:.4f}")
    if total != counted:
        print(f"Warning: counted {counted} of {total} files")


if __name__ == "__main__":
    main()
