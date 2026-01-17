docker run -it \
        --gpus all \
        --user "$(id -u):$(id -g)" \
        -p 1239:9200 \
        --ipc=host \
        --name dllm-dev \
        -v /mnt/hdd/research/dllm-dev:/workspace/dllm-dev \
        dllm-dev:260115 \
    bash
