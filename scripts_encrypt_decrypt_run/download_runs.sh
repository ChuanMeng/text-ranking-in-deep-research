hf download grill-lab/browsecomp-plus-runs \
  --repo-type dataset \
  --include="encrypted_runs.tar.gz" \
  --local-dir-use-symlinks False

tar -xzvf encrypted_runs.tar.gz -C 