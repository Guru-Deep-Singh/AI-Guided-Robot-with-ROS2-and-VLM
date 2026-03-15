import base64
import requests
import json
import re
import os

MODEL_ID = "biggestFudge/qwen2-5-vl-7b-robot-merged-v2"


SYSTEM_PROMPT_FINE_TUNED = (
    "You are a robot navigation controller. "
    "Given a forward camera image and a top-down LiDAR map, "
    "output only a JSON object with a single key 'intent' "
    "with value one of: FORWARD, LEFT, RIGHT, REVERSE, STOP."
)
USER_PROMPT_FINE_TUNED = (
    "IMAGE 1: forward camera view. "
    "IMAGE 2: top-down LiDAR map (robot = cyan dot at centre facing up, "
    "rings = 4m/8m/12m, white dots = obstacles). "
    "What is the next driving action?"
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(raw: str) -> dict:
    """
    Same robust JSON extraction as llm_driver_node_lidar._extract_json.
    Handles markdown code fences, <think> blocks, and plain JSON.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response")

    # Strip <think>...</think> blocks (some local models include these)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Strip markdown code fences ```json ... ``` or ``` ... ```
    raw = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw, flags=re.DOTALL).strip()

    # Try to parse the whole thing as JSON
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract intent or action key directly
    for key in ("intent", "action"):
        key_match = re.search(
            rf'"{key}"\s*:\s*"(FORWARD|LEFT|RIGHT|REVERSE|STOP)"', raw
        )
        if key_match:
            return {key: key_match.group(1)}

    raise ValueError(f"No JSON found in response: {raw[:200]}")


def test_qwen(image_path: str, lidar_path: str = None):
    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        return

    cam_b64 = encode_image(image_path)

    url     = "http://134.199.204.182:8001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Fine-tuned model: exact training prompt, both images
    system_prompt = SYSTEM_PROMPT_FINE_TUNED
    user_content  = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
    ]
    # Add LiDAR image if provided
    if lidar_path and os.path.exists(lidar_path):
        lidar_b64 = encode_image(lidar_path)
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lidar_b64}"}}
        )
    else:
        print("Note: no LiDAR image provided — sending camera only.")
    user_content.append({"type": "text", "text": USER_PROMPT_FINE_TUNED})

    payload = {
        "model":       MODEL_ID,
        "messages":    [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens":  64,
        "temperature": 0.1,
    }

    print(f"Sending request to {url} ...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result      = response.json()
        raw_content = result["choices"][0]["message"]["content"]

        print("\n--- Raw response ---")
        print(raw_content)

        print("\n--- Parsed navigation decision ---")
        try:
            nav    = extract_json(raw_content)
            intent = nav.get('intent') or nav.get('action', '?')
            print(f"  INTENT : {intent}")
            # Print any extra fields if present (general model)
            for key in ('left_column', 'center_column', 'right_column',
                        'yellow_line', 'reasoning'):
                if key in nav:
                    print(f"  {key:<15}: {nav[key]}")
        except ValueError as e:
            print(f"  Parse error: {e}")

        print(f"\nInference time: {result['usage']['inference_seconds']}s")

    except requests.exceptions.Timeout:
        print("Error: request timed out (model may still be loading).")
    except requests.exceptions.HTTPError:
        if response.status_code == 429:
            print("Error 429: server busy — retry after current frame is processed.")
        else:
            print(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    CAMERA_IMAGE = "./test_finetuned.png"
    LIDAR_IMAGE  = "./test_finetuned_lidar.png"   # optional — set to None if you don't have one

    # Download a sample road image if none exists locally
    if not os.path.exists(CAMERA_IMAGE):
        print("Downloading a sample road image for testing...")
        img_data = requests.get(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/"
            "Asphalt_road_texture.jpg/640px-Asphalt_road_texture.jpg"
        ).content
        with open(CAMERA_IMAGE, "wb") as f:
            f.write(img_data)
        print(f"Saved to {CAMERA_IMAGE}")

    test_qwen(CAMERA_IMAGE, lidar_path=LIDAR_IMAGE)