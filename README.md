# Revisiting Text Ranking in Deep Research

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=text-ranking-in-deep-research)

This repository contains the code for the paper [Revisiting Text Ranking in Deep Research](https://arxiv.org/abs/2602.21456).

In this work, we reproduce a comprehensive set of text ranking methods in the context of deep research. Specifically, we investigate the performance of **2 deep research agents** accessing **5 retrievers** and **3 re-rankers**.
The experiments use the agents [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) and [GLM-4.7-Flash (30B)](https://huggingface.co/zai-org/GLM-4.7-Flash).
Retrievers are [BM25](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/okapi_trec3.pdf), [SPLADE-v3](https://huggingface.co/naver/splade-v3), [RepLLaMA](https://huggingface.co/castorini/repllama-v1-7b-lora-passage), [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), and [ColBERTv2](https://huggingface.co/colbert-ir/colbertv2.0).
Re-rankers are [monoT5-3B](https://huggingface.co/castorini/monot5-3b-msmarco), [RankLLaMA-7B](https://huggingface.co/castorini/rankllama-v1-7b-lora-passage), and [Rank1-7B](https://huggingface.co/jhu-clsp/rank1-7b).


## Resources released

To support reproducibility and further analysis, this repository releases:

- Our newly constructed BrowseComp-Plus **passage corpus**, available for download [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-passage-corpus).
- All indexes of retrievers, available for download [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-indexes).
- Full execution trajectory data (including agent reasoning and search traces) for all runs in our paper, available for download [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-runs). Note that the trajectory data is released in encrypted form and can be decrypted locally using the provided scripts. 

## Repository Structure

This repository is organised into the following sections:
- [1. Prerequisites](#sec-prereq)
- [2. Quick start](#sec-quickstart)
- [3. Reproducing results](#sec-reproduce)
  - [3.1 RQ1: Retrievers on passage and document corpora](#sec-rq1)
  - [3.2 RQ2: Re-rankers](#sec-rq2)
  - [3.3 RQ3: Query-to-question (Q2Q) reformulation](#sec-rq3)
- [4. Passage corpus construction](#sec-data)
- [5. Indexing](#sec-index)
  - [5.1 Passage corpus](#sec-index-psg)
  - [5.2 Document corpus](#sec-index-doc)
- [6. Decrypting encrypted runs](#sec-decrypt)
- [7. Contact](#sec-contact)
- [8. Citation](#sec-citation)

<a id="sec-prereq"></a>
## 1. Prerequisites

### 1.1 Environment

We use different environments for (i) hosting local vLLM servers and (ii) running the deep-research agents.

For hosting local vLLM servers for **gpt-oss-20b** and **GLM-4.7-Flash (30B)**:
```bash
uv venv vllm_server --python 3.12
source vllm_server/bin/activate

uv pip install "vllm==0.15.0"
uv pip install -U transformers
uv pip install "numpy==2.2.2" "numba==0.61.2" "llvmlite==0.44.0"
```

For running the agents, we follow the environment provided in the [BrowseComp-Plus repository](https://github.com/texttron/BrowseComp-Plus).
Please follow the installation instructions in that repository.
After installation, activate the environment using:
```bash
source .venv/bin/activate
```

We use [PyLate](https://github.com/lightonai/pylate) to run [ColBERTv2](https://huggingface.co/colbert-ir/colbertv2.0)
, which conflicts with the default BrowseComp-Plus environment. Therefore, we create a separate environment for PyLate:
```bash
uv venv pylate --python 3.10
source pylate/bin/activate

uv pip install pylate
uv pip install pytrec_eval
uv pip install pyserini
uv pip install peft
uv pip install qwen_omni_utils
uv pip install rich
```

### 1.2 Data downloading

Please follow the instructions in the
[BrowseComp-Plus repository](https://github.com/texttron/BrowseComp-Plus)
to download the dataset and generate the decrypted data.
This process will produce the decrypted dataset at
`./data/browsecomp_plus_decrypted.jsonl`, the query file at
`./topics-qrels/queries.tsv`, and two relevance-judgment files at
`./topics-qrels/qrel_evidence.txt` and `./topics-qrels/qrel_golds.txt`.

Rename the query file as follows:
```bash
mv ./topics-qrels/queries.tsv ./topics-qrels/queries-all.tsv
```

#### Download the passage corpus
Download the BrowseComp-Plus passage corpus from [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-passage-corpus):
```bash
bash ./scripts_build_index/download_passage.sh
```
Passage files will be put in the `./data/browsecomp-plus-passage/` directory under the repository root.
If you prefer to construct the passage corpus yourself, follow Section [4](#sec-data).

#### Download retrieval indices
Download all pre-built retriever indices from [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-indexes):
```bash
bash ./scripts_build_index/download_indexes.sh
```
All indexes will be put in the `indexes/` directory under the repository root.
If you prefer to build the indices yourself, follow Section [5](#sec-index).


<a id="sec-quickstart"></a>
## 2. Quick start

We begin by replicating the best-performing configurations reported in our paper: the gpt-oss-20b agent using a pipeline with **BM25** as the retriever and **monoT5-3B** as the re-ranker (with a re-ranking depth of 50) on our passage corpus, which achieves a Recall of **0.716** and an answer accuracy of **0.689**.

First, launch a local gpt-oss-20b server using vLLM:
```bash
source vllm_server/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve openai/gpt-oss-20b \
--tool-call-parser openai --enable-auto-tool-choice \
--host 0.0.0.0 --port 8000
```
Then, run the agent with BM25 + monoT5:
```bash
source .venv/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python search_agent/oss_client.py \
  --model openai/gpt-oss-20b \
  --model-url http://localhost:8000/v1 \
  --searcher-type bm25 \
  --index-path ./indexes/index.bm25.passage/ \
  --output-dir ./runs/gpt-oss-20b-high/queries-all.bm25-d50-monot5-3b-msmarco-k5.passage \
  --reasoning-effort high --max-tokens 40000 --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT \
  --snippet-max-tokens 512 \
  --query ./topics-qrels/queries-all.tsv \
  --reranking_depth 50 --k 5 \
  --reranker-type monot5 \
  --monot5-model castorini/monot5-3b-msmarco --monot5-tokenizer castorini/monot5-3b-msmarco \
  --monot5-batch-size 8 \
  --num-threads 10 

# the Max-P strategy, which maps retrieved passages to documents by assigning each document the maximum score among its retrieved passages
python psg2doc.py \
  --input_json_dir ./runs/gpt-oss-20b-high/queries-all.bm25-d50-monot5-3b-msmarco-k5.passage \
  --output_json_dir ./runs/gpt-oss-20b-high/queries-all.bm25-d50-monot5-3b-msmarco-k5.passage-psg2doc
```

Last, run the following commands to do evaluation:
```bash
# Search calls, Recall, and answer accuracy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
python scripts_evaluation/evaluate_run.py \
--input_dir ./runs/gpt-oss-20b-high/queries-all.bm25-d50-monot5-3b-msmarco-k5.passage-psg2doc \
--ground_truth ./data/browsecomp_plus_decrypted.jsonl \
--qrel_evidence ./topics-qrels/qrel_evidence.txt \
--tensor_parallel_size 2

# completion rate
python scripts_evaluation/count_complete.py \
--input_dir ./runs/gpt-oss-20b-high/queries-all.bm25-d50-monot5-3b-msmarco-k5.passage-psg2doc
```

<a id="sec-reproduce"></a>
## 3. Reproducing results

Before running the experiments, ensure that the required retrieval indices are available.  
You may either download our pre-built indices or build them by following the instructions in Section [5](#sec-index).

Next, launch an agent server required for the experiments.
Launch **gpt-oss-20b**:
```bash
source vllm_server/bin/activate
bash ./run_scripts/server_gpt-oss-20b.sh
```
Launch **GLM-4.7-Flash**:
```bash
source vllm_server/bin/activate
bash ./run_scripts/server_GLM-4.7-Flash.sh
```
You may change the port value in the server scripts if needed.
Make sure that the `MODEL_URL` specified in each experiment script (e.g., `MODEL_URL=http://localhost:8000/v1`) points to the correct server endpoint you intend to use.

<a id="sec-rq1"></a>
### 3.1 RQ1: Retrievers on passage and document corpora

All scripts assume that the corresponding indices have been built and the agent server is already running.

#### gpt-oss-20b

Run the **gpt-oss-20b** agent with **BM25** on  
(i) the passage corpus,  
(ii) the document corpus, and  
(iii) the document corpus with the full-document reader enabled:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.bm25-k5.passage.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.bm25-k5.document.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.bm25-k5.document.getdoc.sh
```

Run the **gpt-oss-20b** agent with **SPLADE-v3** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.splade-v3-k5.passage.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.splade-v3-k5.document.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.splade-v3-k5.document.getdoc.sh
```

Run the **gpt-oss-20b** agent with **RepLLaMA** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.repllama-v1-7b-lora-passage-k5.passage.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.repllama-v1-7b-lora-passage-k5.document.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.repllama-v1-7b-lora-passage-k5.document.getdoc.sh
```

Run the **gpt-oss-20b** agent with **Qwen3-Embedding-8B** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.qwen3-embedding-8b-k5.passage.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.qwen3-embedding-8b-k5.document.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.qwen3-embedding-8b-k5.document.getdoc.sh
```

Run the **gpt-oss-20b** agent with **ColBERTv2** on the three settings:
```bash
source pylate/bin/activate
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.colbertv2.0-k5.passage.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.colbertv2.0-k5.document.sh
bash ./run_scripts/rq1/gpt-oss-20b.queries-all.colbertv2.0-k5.document.getdoc.sh
```

#### GLM-4.7-Flash

Run the **GLM-4.7-Flash** agent with **BM25** on  
(i) the passage corpus,  
(ii) the document corpus, and  
(iii) the document corpus with the full-document reader enabled:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.bm25-k5.passage.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.bm25-k5.document.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.bm25-k5.document.getdoc.sh
```

Run the **GLM-4.7-Flash** agent with **SPLADE-v3** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.splade-v3-k5.passage.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.splade-v3-k5.document.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.splade-v3-k5.document.getdoc.sh
```

Run the **GLM-4.7-Flash** agent with **RepLLaMA** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.repllama-v1-7b-lora-passage-k5.passage.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.repllama-v1-7b-lora-passage-k5.document.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.repllama-v1-7b-lora-passage-k5.document.getdoc.sh
```

Run the **GLM-4.7-Flash** agent with **Qwen3-Embedding-8B** on the three settings:
```bash
source .venv/bin/activate
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.qwen3-embedding-8b-k5.passage.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.qwen3-embedding-8b-k5.document.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.qwen3-embedding-8b-k5.document.getdoc.sh
```

Run the **GLM-4.7-Flash** agent with **ColBERTv2** on the three settings:
```bash
source pylate/bin/activate
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.colbertv2.0-k5.passage.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.colbertv2.0-k5.document.sh
bash ./run_scripts/rq1/GLM-4.7-Flash.queries-all.colbertv2.0-k5.document.getdoc.sh
```


<a id="sec-rq2"></a>
### 3.2 RQ2: Re-rankers

All scripts assume that the corresponding indices have been built and the agent server is already running.

#### gpt-oss-20b

Run the **gpt-oss-20b** agent using ranking pipelines where **BM25**, **SPLADE-v3**, or **Qwen3-Embedding-8B** are used as the retrievers, and **monoT5-3B**, **RankLLaMA-7B** or **Rank1-7B** are used as the re-rankers, on the passage corpus.
The re-ranking depth can be modified by changing the `RERANKING_DEPTH` variable in each script.
```bash
source .venv/bin/activate

bash ./run_scripts/rq2/gpt-oss-20b.queries-all.bm25-monot5-3b-msmarco-k5.passage.sh
bash ./run_scripts/rq2/gpt-oss-20b.queries-all.bm25-rankllama-v1-7b-lora-passage-k5.passage.sh

bash ./run_scripts/rq2/gpt-oss-20b.queries-all.splade-v3-monot5-3b-msmarco-k5.passage.sh
bash ./run_scripts/rq2/gpt-oss-20b.queries-all.splade-v3-rankllama-v1-7b-lora-passage-k5.passage.sh

bash ./run_scripts/rq2/gpt-oss-20b.queries-all.qwen3-embedding-8b-monot5-3b-msmarco-k5.passage.sh
bash ./run_scripts/rq2/gpt-oss-20b.queries-all.qwen3-embedding-8b-rankllama-v1-7b-lora-passage-k5.passage.sh
```

For Rank1-7B, to improve efficiency we host the model on a local server.
First launch the Rank1 server:
```bash
bash server_rank1-7b.sh
```
Then run the pipelines with Rank1.
Ensure that the `RANK1_MODEL_URL` variable in each run script matches the Rank1 server endpoint (e.g., `http://localhost:8001/v1`).
```bash
source .venv/bin/activate

bash ./run_scripts/rq2/gpt-oss-20b.queries-all.bm25-rank1-7b-k5.passage.sh
bash ./run_scripts/rq2/gpt-oss-20b.queries-all.splade-v3-rank1-7b-k5.passage.sh
bash ./run_scripts/rq2/gpt-oss-20b.queries-all.qwen3-embedding-8b-rank1-7b-k5.passage.sh
```


#### GLM-4.7-Flash
Run the **GLM-4.7-Flash** agent using ranking pipelines where **BM25** or **SPLADE-v3** is used as the retriever, and **monoT5-3B** or **RankLLaMA-7B** is used as the re-ranker, on the passage corpus.
The re-ranking depth can be modified by changing the `RERANKING_DEPTH` variable in each script.
```bash
source .venv/bin/activate

bash ./run_scripts/rq2/GLM-4.7-Flash.queries-all.bm25-monot5-3b-msmarco-k5.passage.sh
bash ./run_scripts/rq2/GLM-4.7-Flash.queries-all.bm25-rankllama-v1-7b-lora-passage-k5.passage.sh

bash ./run_scripts/rq2/GLM-4.7-Flash.queries-all.splade-v3-monot5-3b-msmarco-k5.passage.sh
bash ./run_scripts/rq2/GLM-4.7-Flash.queries-all.splade-v3-rankllama-v1-7b-lora-passage-k5.passage.sh
```

<a id="sec-rq3"></a>
### 3.3 RQ3: Query-to-question (Q2Q) reformulation

All scripts assume that the corresponding indices have been built and the agent server is already running.

Run the **gpt-oss-20b** agent using **Q2Q-rewritten queries** with **BM25**, **SPLADE-v3**, or **Qwen3-Embedding-8B** as the retriever on the **passage corpus**:
```bash
source .venv/bin/activate

bash ./run_scripts/rq3/gpt-oss-20b.queries-all.q2q-context-bm25-k5.passage.sh
bash ./run_scripts/rq3/gpt-oss-20b.queries-all.q2q-context-splade-v3-k5.passage.sh
bash ./run_scripts/rq3/gpt-oss-20b.queries-all.q2q-context-qwen3-embedding-8b-k5.passage.sh
```

<a id="sec-data"></a>
## 4. Passage corpus construction

We construct the passage corpus using the same segmentation pipeline employed in previous **TREC CAsT** tracks.

First, clone the repository of
[trec-cast-tools](https://github.com/grill-lab/trec-cast-tools),
and run the following command to segment documents into passages.
The generated passage files will be saved in the `./data/jsonlines` directory:

```bash
python -u ./corpus_processing/main.py \
  --skip_process_kilt \
  --skip_process_marco_v2 \
  --skip_process_wapo \
  --browsecomp_plus_collection Tevatron/browsecomp-plus-corpus \
  --output_type jsonlines \
  --output_dir ./data/
```
Next, return to the working directory of this repository and preprocess the generated passage file (e.g., assign passage IDs to each document and compute passage lengths).
The following command produces two passage-corpus files: one in the format required by [Pyserini](https://github.com/castorini/pyserini) and the other in the format required by [Tevatron](https://github.com/texttron/tevatron):
```bash
python -u passage_corpus_preprocess.py \
--input_dir ./data/jsonlines \
--output_dir ./data/browsecomp-plus-passage
```

<a id="sec-index"></a>
## 5. Indexing 

This section describes how to build retrieval indices for both the **passage corpus** constructed in this work and the original **document corpus**.
All commands below assume execution from the repository root unless otherwise specified.

<a id="sec-index-psg"></a>
### 5.1 Passage corpus

#### BM25
```bash
source .venv/bin/activate

python -m pyserini.index.lucene \
--collection JsonCollection \
--input ./data/browsecomp-plus-passage/browsecomp-plus-passage-pyserini/ \
--index ./indexes/index.bm25.passage \
--generator DefaultLuceneDocumentGenerator \
--threads 16 \
--storePositions --storeDocvectors --storeRaw
```

#### SPLADE-v3
Regarding SPLADE-v3, first ensure that [Tevatron](https://github.com/texttron/tevatron) is installed, and switch to the Tevatron directory.
Run the following commands to encode the passage corpus:
```bash
source .venv/bin/activate

mkdir -p ./indexes/encoding.splade-v3.passage

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python ./examples/splade/encode_splade.py \
  --model_name_or_path naver/splade-v3 \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 512 \
  --per_device_eval_batch_size 128 \
  --dataset_path ./data/browsecomp-plus-passage/browsecomp-plus-passage-tevatron.jsonl \
  --dataset_config jsonl \
  --output_dir ./indexes/encoding.splade-v3.passage \
  --encode_output_path ./indexes/encoding.splade-v3.passage/corpus.jsonl
```
After encoding, build the Lucene index using [Pyserini](https://github.com/castorini/pyserini):
```bash
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ./indexes/encoding.splade-v3.passage/ \
  --index ./indexes/index.splade-v3.passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### RepLLaMA
```bash
source .venv/bin/activate

mkdir ./indexes/index.repllama-v1-7b-lora-passage.passage
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 32 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --dataset_path ./data/browsecomp-plus-passage/browsecomp-plus-passage-tevatron.jsonl \
  --dataset_config jsonl \
  --encode_output_path ./indexes/index.repllama-v1-7b-lora-passage.passage/corpus.pkl 
```

#### Qwen3-Embedding-8B
```bash
source .venv/bin/activate

mkdir ./indexes/index.qwen3-embedding-8b.passage
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.retriever.driver.encode \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --dataset_path ./data/browsecomp-plus-passage/browsecomp-plus-passage-tevatron.jsonl \
  --dataset_config jsonl \
  --encode_output_path ./indexes/index.qwen3-embedding-8b.passage/corpus.pkl \
  --passage_max_len 512 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --per_device_eval_batch_size 64 \
  --fp16
```

#### ColBERTv2
```bash
source pylate/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
 python colbert.py \
--model_name colbert-ir/colbertv2.0 \
--corpus_path ./data/browsecomp-plus-passage/browsecomp-plus-passage-tevatron.jsonl \
--index_folder ./indexes/index.colbertv2.0-d512.passage \
--batch_size 2048 \
--max_seq_length 512 \
--document_length 512 \
--query_length 512 \
--chunk_size 100000 --shard_count 4 \
--build_index 
```

<a id="sec-index-doc"></a>
### 5.2 Document corpus 

We use the pre-built BM25 index and the Qwen3-Embedding-8B index for the document corpus released by the [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) authors.
Here we show how to index the document corpus using SPLADE-v3, RepLLaMA and ColBERTv2.

#### SPLADE-v3
```bash
source .venv/bin/activate
# First ensure that Tevatron is installed and switch to the Tevatron directory.
python ./examples/splade/encode_splade.py \
  --model_name_or_path naver/splade-v3 \
  --tokenizer_name bert-base-uncased \
  --fp16 \
  --passage_max_len 512 \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/browsecomp-plus-corpus \
  --output_dir ./indexes/index.splade-v3.document \
  --encode_output_path ./indexes/encoding.splade-v3.document/corpus.jsonl

python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ./indexes/encoding.splade-v3.document/ \
  --index ./indexes/index.splade-v3.document \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --impact --pretokenized
```

#### RepLLaMA
```bash
source .venv/bin/activate

mkdir ./indexes/index.repllama-v1-7b-lora-passage.document

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m tevatron.retriever.driver.encode \
  --output_dir=./temp \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
  --lora \
  --query_prefix "query: " \
  --passage_prefix "passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 64 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --dataset_name Tevatron/browsecomp-plus-corpus \
  --encode_output_path ./indexes/index.repllama-v1-7b-lora-passage.document/corpus.pkl 
```

#### ColBERTv2
```bash
source pylate/bin/activate

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
 python colbert.py \
--model_name colbert-ir/colbertv2.0 \
--corpus_path Tevatron/browsecomp-plus-corpus \
--index_folder ./indexes/index.colbertv2.0-d512.document \
--batch_size 2048 \
--max_seq_length 512 \
--document_length 512 \
--query_length 512 \
--shard_count 4 \
--build_index 
```

<a id="sec-decrypt"></a>
## 6. Decrypting encrypted runs

Download the encrypted run files from [here](https://huggingface.co/datasets/grill-lab/browsecomp-plus-runs) and decrypt them:
```bash
bash ./scripts_encrypt_decrypt_run/download_runs.sh
bash ./scripts_encrypt_decrypt_run/decrypt_runs.sh
```
This script will decrypt all run files locally and generate the corresponding plaintext execution traces.


<a id="sec-contact"></a>
## 7. Contact
If you have any questions or suggestions, please contact us at:
- [Chuan Meng](https://chuanmeng.github.io/): chuan.meng@ed.ac.uk
- [Litu Ou](https://leonard907.github.io/): litu.ou@ed.ac.uk


<a id="sec-citation"></a>
## 8. Citation
If you find this work useful, please cite:
```
@article{meng2026revisiting,
      title={Revisiting Text Ranking in Deep Research}, 
      author={Meng, Chuan and Ou, Litu and MacAvaney, Sean and Dalton, Jeff},
      year={2026},
      journal={arXiv preprint arXiv:2602.21456}
}
```
