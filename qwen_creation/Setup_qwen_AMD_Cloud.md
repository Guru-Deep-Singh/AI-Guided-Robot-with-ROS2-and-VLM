
# Steps to host Qwen2.5-VL on AMD Cloud

- Expose the 8001 port from the AMD Cloud instance to the internet.
``` bash
sudo ufw allow 8001/tcp
sudo ufw status
```

- Create a folder and paste the code to qwen_vl_server.py and Dockerfile.
``` bash
mkdir qwen_vl
cd qwen_vl

# Paste the code to qwen_vl_server.py
# Paste the Dockerfile


docker build --no-cache -t qwen-vl:rocm .

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g \
  -p 8001:8000 \
  -e MODEL_ID="Qwen/Qwen2.5-VL-72B-Instruct" \
  qwen-vl:rocm