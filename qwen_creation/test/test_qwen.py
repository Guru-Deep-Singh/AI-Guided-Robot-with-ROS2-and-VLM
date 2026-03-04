import base64
import requests
import json
import os

SYSTEM_PROMPT = """You are a robot navigation controller. The robot is ~2.55 m long, ~1.27 m wide, driving on a gray asphalt road.

ROAD MARKINGS:
- YELLOW center line: use this to detect road direction and stay centered.
- WHITE edge lines: road boundaries — never cross them.
- Green surface: off-road, avoid.

ANALYZE IN ORDER:

1. COLUMNS — describe what you see in LEFT / CENTER / RIGHT thirds:
   Use only: asphalt | cone | white_line | yellow_line | green | wall | unclear

2. YELLOW LINE DIRECTION — trace the yellow center line from bottom to top:
   - Drifts left in top half → road curves LEFT
   - Drifts right in top half → road curves RIGHT
   - Stays centered → STRAIGHT

3. ACTION — pick exactly one using this priority:
   - STOP: all three columns fully blocked (last resort only)
   - LEFT: road curves left, OR obstacle in center with left side clear
   - RIGHT: road curves right, OR obstacle in center with right side clear
   - FORWARD: center clear, yellow line straight and centered
   - If near white edge: steer away from it

OUTPUT — respond with ONLY this JSON, no extra text:
{
  "left_column": "<what you see>",
  "center_column": "<what you see>",
  "right_column": "<what you see>",
  "yellow_line": "drifts_left | drifts_right | centered",
  "action": "FORWARD | LEFT | RIGHT | STOP",
  "reasoning": "<max 8 words>"
}"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_qwen(image_path: str, user_prompt: str = "What direction should the robot move?"):
    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        return

    base64_image = encode_image(image_path)

    url     = "http://165.245.128.132:8001/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0.1,   # low temp = more deterministic navigation decisions
    }

    print(f"Sending request to Qwen2.5-VL at {url} ...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result  = response.json()
        content = result["choices"][0]["message"]["content"]

        print("\n--- Raw response ---")
        print(content)

        # Try to parse as JSON and pretty-print the navigation decision
        print("\n--- Parsed navigation decision ---")
        try:
            nav = json.loads(content)
            print(f"  Left   : {nav.get('left_column')}")
            print(f"  Center : {nav.get('center_column')}")
            print(f"  Right  : {nav.get('right_column')}")
            print(f"  Yellow : {nav.get('yellow_line')}")
            print(f"  ACTION : {nav.get('action')}")
            print(f"  Reason : {nav.get('reasoning')}")
        except json.JSONDecodeError:
            print("  (response was not valid JSON — check the system prompt or model output)")

        print(f"\nInference time: {result['usage']['inference_seconds']}s")

    except requests.exceptions.Timeout:
        print("Error: request timed out (model may still be loading).")
    except requests.exceptions.HTTPError:
        if response.status_code == 429:
            print("Error 429: server busy — non-blocking lock dropped the frame.")
        else:
            print(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    IMAGE_PATH = "./test.png"

    # Download a sample road image if none exists locally
    if not os.path.exists(IMAGE_PATH):
        print("Downloading a sample road image for testing...")
        img_data = requests.get(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/"
            "Asphalt_road_texture.jpg/640px-Asphalt_road_texture.jpg"
        ).content
        with open(IMAGE_PATH, "wb") as f:
            f.write(img_data)
        print(f"Saved to {IMAGE_PATH}")

    test_qwen(IMAGE_PATH)