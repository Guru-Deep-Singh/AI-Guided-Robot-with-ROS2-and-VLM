FROM rocm/7.0:rocm7.0_pytorch_training_instinct_20250915

WORKDIR /app
COPY qwen_vl_server.py /app/qwen_vl_server.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pillow \
    "transformers>=4.49.0" \
    accelerate \
    qwen-vl-utils \
    huggingface_hub

ENV MODEL_ID="Qwen/Qwen2.5-VL-72B-Instruct"
ENV MAX_NEW_TOKENS=256
ENV MIN_PIXELS=200704
ENV MAX_PIXELS=401408
ENV PORT=8000

EXPOSE 8000

CMD ["python3", "qwen_vl_server.py"]