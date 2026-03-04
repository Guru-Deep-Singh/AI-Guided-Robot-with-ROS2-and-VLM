"""
qwen_vl_server.py

Clean, single-file FastAPI server for Qwen2.5-VL-7B-Instruct.
Exposes an OpenAI-compatible POST /v1/chat/completions endpoint.

Designed for robotics / real-time use:
  - Model loaded once at startup
  - Non-blocking lock: returns HTTP 429 immediately if busy (drop-frame semantics)
  - Stateless: no session memory between requests

Usage:
  docker run ... -e MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct" ...

Request (OpenAI-like):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": [
        {"type": "text",      "text": "What direction?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
  ],
  "max_tokens": 256,
  "temperature": 0.1
}
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

# Qwen2.5-VL uses its own processor from the transformers library
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
PORT           = int(os.getenv("PORT", "8000"))

# Min/max pixels control the image resolution fed to the vision encoder.
# Lower = faster inference. For a robot camera at 640x480, these defaults work well.
MIN_PIXELS = int(os.getenv("MIN_PIXELS", str(256 * 28 * 28)))   # ~200k
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(512 * 28 * 28)))   # ~400k

# Non-blocking lock — drop the frame if the model is still generating
_LOCK = threading.Lock()

app = FastAPI(title="Qwen2.5-VL Robot Vision Server", version="1.0.0")


# ── Pydantic request models (OpenAI-compatible) ───────────────────────────────

class ImageURL(BaseModel):
    url: str  # must be a data: URL (base64)

class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str

class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL

ContentPart = Union[ContentPartText, ContentPartImage]

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart], None] = None

class ChatRequest(BaseModel):
    model:          Optional[str]   = Field(default=None)
    messages:       List[Message]
    max_tokens:     Optional[int]   = None
    max_new_tokens: Optional[int]   = None
    temperature:    Optional[float] = 0.1
    top_p:          Optional[float] = 0.9
    stream:         Optional[bool]  = False


# ── Model loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_model(model_id: str):
    """
    Load Qwen2.5-VL model + processor once and cache forever.
    Qwen2.5-VL has full first-class HuggingFace support — no patching needed.
    """
    print(f"Loading {model_id} …")

    dtype     = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto"       if torch.cuda.is_available() else None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Pass min/max pixels so the processor resizes images appropriately
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    print(f"✓ {model_id} loaded on {next(model.parameters()).device}")
    return model, processor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(data_url: str) -> Image.Image:
    """Decode a base64 data: URL into a PIL Image."""
    if not data_url.startswith("data:"):
        raise ValueError("image_url.url must be a data: URL (base64 encoded).")
    try:
        _, b64 = data_url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e


def _parse_messages(messages: List[Message]):
    """
    Convert OpenAI-style messages into:
      - qwen_messages: list of dicts for the Qwen chat template
      - images: list of PIL Images (in order of appearance)
    """
    qwen_messages = []
    images: List[Image.Image] = []

    for msg in messages:
        if isinstance(msg.content, str):
            # Plain text message (common for system prompt)
            qwen_messages.append({
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}],
            })

        elif isinstance(msg.content, list):
            content_parts = []
            for part in msg.content:
                if isinstance(part, ContentPartText):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ContentPartImage):
                    img = _decode_image(part.image_url.url)
                    images.append(img)
                    # Qwen2.5-VL uses {"type": "image"} as the placeholder
                    content_parts.append({"type": "image"})
            qwen_messages.append({"role": msg.role, "content": content_parts})

    return qwen_messages, images


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(
    messages: List[Message],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model, processor = _load_model(MODEL_ID)

    qwen_messages, images = _parse_messages(messages)

    if not images:
        raise HTTPException(status_code=400, detail="No image found in request.")

    # Apply Qwen chat template — this handles the special image tokens automatically
    text_prompt = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process text + images together
    inputs = processor(
        text=[text_prompt],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    # Non-blocking lock: return 429 immediately if another request is generating
    acquired = _LOCK.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Model busy — retry after current frame is processed.",
        )

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=float(temperature) if temperature > 0 else None,
                top_p=float(top_p)             if temperature > 0 else None,
                use_cache=True,
            )
    finally:
        _LOCK.release()

    # Decode only the newly generated tokens (strip the prompt)
    input_len = inputs["input_ids"].shape[1]
    generated  = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── Startup warmup ────────────────────────────────────────────────────────────

@app.on_event("startup")
def _warmup():
    _load_model(MODEL_ID)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported.")

    # Resolve max_new_tokens
    max_new_tokens = req.max_new_tokens or req.max_tokens or MAX_NEW_TOKENS
    max_new_tokens = max(1, int(max_new_tokens))

    temperature = float(req.temperature or 0.1)
    top_p       = float(req.top_p or 0.9)

    t0   = time.time()
    text = _infer(req.messages, max_new_tokens, temperature, top_p)
    elapsed = round(time.time() - t0, 3)

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     None,
            "completion_tokens": None,
            "total_tokens":      None,
            "inference_seconds": elapsed,
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Single worker = single model copy in VRAM
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)