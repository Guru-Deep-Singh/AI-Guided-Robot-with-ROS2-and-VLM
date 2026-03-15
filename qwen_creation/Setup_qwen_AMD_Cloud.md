# Steps to host Qwen2.5-VL on AMD Cloud

## Prerequisites

- AMD Cloud instance with MI300X
- Docker installed with ROCm support
- HuggingFace token with read access (get from https://huggingface.co/settings/tokens)

---

## Setup

**Expose port 8001:**
```bash
sudo ufw allow 8001/tcp
sudo ufw status
```

**Create workspace and add files:**
```bash
mkdir qwen_vl
cd qwen_vl
# Paste qwen_vl_server.py (or qwen_vl_server_finetuned.py for the fine-tuned model) and Dockerfile into this folder
```

**Build the Docker image (one-time):**
```bash
docker build --no-cache -t qwen-vl:rocm .
```

---

## Running the 72B general model

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g \
  -p 8001:8000 \
  -e MODEL_ID="Qwen/Qwen2.5-VL-72B-Instruct" \
  qwen-vl:rocm
```

---

## Running the fine-tuned 7B robot model

> Ensure you configure the container or Dockerfile to use `qwen_vl_server_finetuned.py` as your server script if specifically setting up the fine-tuned infrastructure.

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g \
  -p 8001:8000 \
  -e MODEL_ID="biggestFudge/qwen2-5-vl-7b-robot-merged-v2" \
  -e MIN_PIXELS="200704" \
  -e MAX_PIXELS="200704" \
  qwen-vl:rocm
```

> **MIN_PIXELS and MAX_PIXELS must be 200704 (= 448×448)** — this matches the
> resolution used during fine-tuning. Using different values will cause the model
> to see a different number of image tokens than it was trained on and will
> degrade performance.

> **HF_TOKEN** is required to download the private merged model from HuggingFace
> on first run. The model is cached inside the container after that.
> Get your token from https://huggingface.co/settings/tokens

---

## ROS2 .env configuration

Update `~/ros-with-ai/.env` to point at the fine-tuned model:

```bash
LLM_BACKEND=local
LOCAL_BASE_URL=http://<your_amd_cloud_ip>:8001/v1
LOCAL_MODEL_TYPE=finetuned
LOCAL_MODEL=biggestFudge/qwen2-5-vl-7b-robot-merged-v2
LOCAL_API_KEY=none
LLM_INTERVAL=1.5
IMAGE_SIZE=320x320
JPEG_QUALITY=85
```

---

## Testing

**Health check:**
```bash
curl http://<your_amd_cloud_ip>:8001/health
```

**Full inference test:**
```bash
python3 test_qwen.py
```

**Expected response time:** under 1 second per call for the 7B model on MI300X
(vs several seconds for the 72B model).

---

## Switching between models

To go back to the 72B model, just stop the container and rerun with the 72B
`MODEL_ID` — no rebuild needed. The image is the same for both models.

| Model | Speed | Use case |
|-------|-------|----------|
| `Qwen/Qwen2.5-VL-72B-Instruct` | ~3-5s/call | General baseline |
| `biggestFudge/qwen2-5-vl-7b-robot-merged-v2` | <1s/call | Fine-tuned robot navigation |