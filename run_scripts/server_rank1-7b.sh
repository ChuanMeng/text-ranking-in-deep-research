CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
vllm serve jhu-clsp/rank1-7b \
--port 9000 --host 0.0.0.0 \
--max-model-len 4000 --trust-remote-code