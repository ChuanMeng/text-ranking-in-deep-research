CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  \
vllm serve openai/gpt-oss-20b \
--tool-call-parser openai --enable-auto-tool-choice \
--host 0.0.0.0 --port 8000