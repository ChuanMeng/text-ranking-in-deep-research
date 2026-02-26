hf download grill-lab/browsecomp-plus-runs \
  --repo-type dataset \
  --include="encrypted_runs.tar.gz" \
  --local-dir .

mkdir -p ./encrypted_runs
tar -xzvf encrypted_runs.tar.gz -C ./encrypted_runs