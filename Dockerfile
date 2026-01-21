FROM nvcr.io/nvidia/tritonserver:24.01-py3
RUN python3 -m pip install -U pip \
 && python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
