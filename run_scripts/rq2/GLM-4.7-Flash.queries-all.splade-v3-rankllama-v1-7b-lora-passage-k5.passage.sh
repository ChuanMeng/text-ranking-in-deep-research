export CUDA_DEVICE_ORDER=PCI_BUS_ID

############################
# GPUs
############################
export GPU_AGENT=0
export GPU_EVAL=1,2

############################
# model
############################
MODEL=zai-org/GLM-4.7-Flash
MODEL_URL=${MODEL_URL:-http://MODEL_SERVER/v1}

# extract short model name (after "/")
MODEL_NAME=${MODEL##*/}

############################
# experiment settings
############################
# RERANKING_DEPTH: depth of reranking; set the number according to your needs
RERANKING_DEPTH=50 

NUM_THREADS=10
K=5
MAX_TOKENS=40000
SNIPPET_MAX_TOKENS=512

QUERY_TEMPLATE=QUERY_TEMPLATE_NO_GET_DOCUMENT

QUERY_FILE=./topics-qrels/queries-all.tsv

INDEX_PATH=./indexes/index.splade-v3.passage
DATASET_NAME=./data/browsecomp-plus-passage/browsecomp-plus-passage-tevatron.jsonl

QUERY_NAME=$(basename "${QUERY_FILE}" .tsv)

if [[ "${INDEX_PATH}" == *.pkl ]]; then
  INDEX_DIR="$(dirname "${INDEX_PATH}")"
else
  INDEX_DIR="${INDEX_PATH%/}"
fi

INDEX_DIR_BASE="$(basename "${INDEX_DIR}")"  
TMP="${INDEX_DIR_BASE#index.}"               
CORPUS_TYPE="${TMP##*.}"                     

OUT_DIR=./runs/${MODEL_NAME}/${QUERY_NAME}.splade-v3-d${RERANKING_DEPTH}-rankllama-v1-7b-lora-passage-k${K}.${CORPUS_TYPE}-psgid
OUT_DOC_DIR=${OUT_DIR}

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
echo "MAX_TOKENS:         ${MAX_TOKENS}"
echo "SNIPPET_MAX_TOKENS: ${SNIPPET_MAX_TOKENS}"
echo "QUERY_TEMPLATE:     ${QUERY_TEMPLATE}"

echo "RERANKING_DEPTH:    ${RERANKING_DEPTH}"
echo "K:                  ${K}"

echo "QUERY_FILE:         ${QUERY_FILE}"
echo "QUERY_NAME:         ${QUERY_NAME}"
echo "INDEX_PATH:         ${INDEX_PATH}"
echo "DATASET_NAME:       ${DATASET_NAME}"
echo "CORPUS_TYPE:        ${CORPUS_TYPE}"

echo "OUT_DIR:            ${OUT_DIR}"
echo "OUT_DOC_DIR:        ${OUT_DOC_DIR}"
echo "=========================================================="

############################
# run agent
############################

CUDA_VISIBLE_DEVICES=${GPU_AGENT} \
python search_agent/oss_client.py \
  --model ${MODEL} \
  --model-url ${MODEL_URL} \
  --searcher-type splade \
  --model-name naver/splade-v3 \
  --tokenizer-name bert-base-uncased \
  --query ${QUERY_FILE} \
  --index-path ${INDEX_PATH} \
  --dataset-name ${DATASET_NAME} \
  --output-dir ${OUT_DIR} \
  --max-tokens ${MAX_TOKENS} \
  --snippet-max-tokens ${SNIPPET_MAX_TOKENS} \
  --query-template ${QUERY_TEMPLATE} \
  --k ${K} \
  --reranking-depth ${RERANKING_DEPTH} \
  --reranker-type rankllama \
  --rankllama-model meta-llama/Llama-2-7b-hf \
  --rankllama-tokenizer meta-llama/Llama-2-7b-hf \
  --rankllama-lora castorini/rankllama-v1-7b-lora-passage \
  --rankllama-query-prefix "query: " \
  --rankllama-passage-prefix "document: " \
  --rankllama-batch-size 8 \
  --num-threads ${NUM_THREADS}

############################
# passage -> document
############################

python psg2doc.py \
  --input_json_dir ${OUT_DIR} \
  --output_json_dir ${OUT_DOC_DIR}

############################
# evaluation
############################

CUDA_VISIBLE_DEVICES=${GPU_EVAL} \
python scripts_evaluation/evaluate_run.py \
  --input_dir ${OUT_DOC_DIR} \
  --ground_truth ./data/browsecomp_plus_decrypted.jsonl \
  --qrel_evidence ./topics-qrels/qrel_evidence.txt \
  --tensor_parallel_size 2

python scripts_evaluation/count_complete.py \
  --input_dir ${OUT_DOC_DIR}