CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
vllm serve zai-org/GLM-4.7-Flash \
--port 8000 \
--host 0.0.0.0 \
--reasoning-parser glm45 \
--tool-call-parser glm47 \
--enable-auto-tool-choice \
--tensor-parallel-size 2