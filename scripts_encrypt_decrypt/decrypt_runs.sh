#!/bin/bash
set -euo pipefail

INPUT_ROOT="./encrypted_runs"
OUTPUT_ROOT="./decrypted_runs"
SCRIPT_PATH="./encrypt_run.py"

# Find all directories containing at least one .json file and process them.
while IFS= read -r dir; do
  rel_path="${dir#${INPUT_ROOT}/}"
  out_dir="${OUTPUT_ROOT}/${rel_path}"
  python3 "${SCRIPT_PATH}" \
    --input-dir "${dir}" \
    --output-dir "${out_dir}" \
    --mode dec
  echo "Decrypted: ${dir} -> ${out_dir}"
done < <(find "${INPUT_ROOT}" -type f -name "*.json" -print0 | xargs -0 -n1 dirname | sort -u)
