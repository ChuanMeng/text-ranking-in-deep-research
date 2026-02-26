export CUDA_DEVICE_ORDER=PCI_BUS_ID

############################
# GPUs
############################
export GPU_AGENT=0
export GPU_EVAL=1,2

############################
# model
############################
MODEL=openai/gpt-oss-20b
MODEL_URL=${MODEL_URL:-http://MODEL_SERVER/v1}

# extract short model name (after "/")
MODEL_NAME=${MODEL##*/}

############################
# experiment settings
############################

MAX_QUERY_LENGTH=512
MAX_DOCUMENT_LENGTH=512

NUM_THREADS=10
REASONING_EFFORT=high
K=5
MAX_TOKENS=40000
SNIPPET_MAX_TOKENS=512

QUERY_TEMPLATE=QUERY_TEMPLATE

QUERY_FILE=./topics-qrels/queries-all.tsv
INDEX_PATH=./indexes/index.colbertv2.0.document
DATASET_NAME=Tevatron/browsecomp-plus-corpus

QUERY_NAME=$(basename "${QUERY_FILE}" .tsv)

if [[ "${INDEX_PATH}" == *.pkl ]]; then
  INDEX_DIR="$(dirname "${INDEX_PATH}")"
else
  INDEX_DIR="${INDEX_PATH%/}"
fi

INDEX_DIR_BASE="$(basename "${INDEX_DIR}")"  
TMP="${INDEX_DIR_BASE#index.}"               
CORPUS_TYPE="${TMP##*.}"                

OUT_DIR=./runs/${MODEL_NAME}-${REASONING_EFFORT}/${QUERY_NAME}.colbertv2.0-q${MAX_QUERY_LENGTH}-d${MAX_DOCUMENT_LENGTH}-k${K}.${CORPUS_TYPE}.getdoc

############################
# print configuration
############################
echo "================ Experiment Configuration ================"
echo "MODEL:              ${MODEL}"
echo "MODEL_NAME:         ${MODEL_NAME}"
echo "MODEL_URL:          ${MODEL_URL}"

echo "GPU_AGENT:          ${GPU_AGENT}"
echo "GPU_EVAL:           ${GPU_EVAL}"

echo "NUM_THREADS:        ${NUM_THREADS}"
echo "REASONING_EFFORT:   ${REASONING_EFFORT}"
echo "K:                  ${K}"
echo "MAX_TOKENS:         ${MAX_TOKENS}"
echo "SNIPPET_MAX_TOKENS: ${SNIPPET_MAX_TOKENS}"
echo "QUERY_TEMPLATE:     ${QUERY_TEMPLATE}"

echo "QUERY_FILE:         ${QUERY_FILE}"
echo "QUERY_NAME:         ${QUERY_NAME}"
echo "INDEX_PATH:         ${INDEX_PATH}"
echo "DATASET_NAME:       ${DATASET_NAME}"
echo "CORPUS_TYPE:        ${CORPUS_TYPE}"

echo "OUT_DIR:            ${OUT_DIR}"
echo "=========================================================="

############################
# run agent
############################

CUDA_VISIBLE_DEVICES=${GPU_AGENT} \
python search_agent/oss_client.py \
  --model ${MODEL} \
  --model-url ${MODEL_URL} \
  --searcher-type colbert \
  --model-name colbert-ir/colbertv2.0 \
  --max-query-length ${MAX_QUERY_LENGTH} \
  --max-document-length ${MAX_DOCUMENT_LENGTH} \
  --query ${QUERY_FILE} \
  --index-path ${INDEX_PATH} \
  --dataset-name ${DATASET_NAME} \
  --output-dir ${OUT_DIR} \
  --max-tokens ${MAX_TOKENS} \
  --snippet-max-tokens ${SNIPPET_MAX_TOKENS} \
  --query-template ${QUERY_TEMPLATE} \
  --reasoning-effort ${REASONING_EFFORT} \
  --k ${K} \
  --num-threads ${NUM_THREADS} \
  --get-document

############################
# evaluation
############################

CUDA_VISIBLE_DEVICES=${GPU_EVAL} \
python scripts_evaluation/evaluate_run.py \
  --input_dir ${OUT_DIR} \
  --ground_truth ./data/browsecomp_plus_decrypted.jsonl \
  --qrel_evidence ./topics-qrels/qrel_evidence.txt \
  --tensor_parallel_size 2

python scripts_evaluation/count_complete.py \
  --input_dir ${OUT_DIR}