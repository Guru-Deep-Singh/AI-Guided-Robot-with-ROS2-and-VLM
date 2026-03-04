import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rwi_interfaces.msg import AgentIntent
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import base64
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import re


load_dotenv()
if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_BACKEND"):
    load_dotenv(os.path.join(os.path.expanduser('~'), 'ros-with-ai', '.env'))

# ------ Backend config -------
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  # "openai" or "local"

if LLM_BACKEND == "local":
    LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")
    LOCAL_API_KEY  = os.getenv("LOCAL_API_KEY", "none")
    if not LOCAL_BASE_URL:
        raise ValueError("LOCAL_BASE_URL must be set in .env when using LLM_BACKEND=local")
    # Strip trailing /chat/completions if someone pasted the full path
    LOCAL_BASE_URL = LOCAL_BASE_URL.rstrip("/")
    if LOCAL_BASE_URL.endswith("/chat/completions"):
        LOCAL_BASE_URL = LOCAL_BASE_URL[: -len("/chat/completions")]
    MODEL  = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
else:
    MODEL  = os.getenv("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------- Prompts -------
SYSTEM_PROMPT_OPENAI = """
You are an autonomous driving system controlling a robot approximately 2.55 m long and 1.27 m wide.

Your goal is to move forward while staying on the gray asphalt road. If a obstacle is detected close to the robot, avoid it and continue moving forward. However, if it is far from the robot, ignore it and continue moving forward.

ROAD LAYOUT:
- The road has a YELLOW center line and WHITE edge lines on both sides.
- You must NEVER cross the WHITE edge lines.
- You MAY cross the YELLOW center line only to avoid an obstacle.
- Green surface is NOT drivable.

STEP 1 — ANALYZE THE IMAGE:
Divide the image into three vertical columns: LEFT third, CENTER third, RIGHT third.
Divide vertically into TOP half (far ahead) and BOTTOM half (close to robot).
For each column report: asphalt, cone, white line, yellow line, green surface, or wall.

STEP 2 — DETECT ROAD DIRECTION using ONLY the YELLOW center line:
- Trace the YELLOW center line from the BOTTOM of the image upward to the TOP.
- In the BOTTOM half, the yellow line is roughly centered. Note where it goes in the TOP half.
- If the yellow line drifts to the LEFT in the TOP half: the road curves LEFT → you must turn LEFT.
- If the yellow line drifts to the RIGHT in the TOP half: the road curves RIGHT → you must turn RIGHT.
- If the yellow line stays centered: road is STRAIGHT → go FORWARD.
- IMPORTANT: Do NOT use the white edge lines to determine curve direction. White lines are edges and can be misleading. Only use the YELLOW center line.

STEP 3 — DECIDE ACTION using this priority order:

1. STOP: Only if ALL three columns are completely blocked. Last resort only.

2. ROAD CURVES (from Step 2, yellow line analysis):
   - Curve LEFT: Action = "LEFT"
   - Curve RIGHT: Action = "RIGHT"
   - Only apply this if you are confident the yellow line is clearly visible and curving.

3. FORWARD: If the yellow line is centered and straight, CENTER column is clear, and you are not near a white edge.

4. OBSTACLE (cone in CENTER column):
   - If LEFT has more clear asphalt: Action = "LEFT"
   - If RIGHT has more clear asphalt: Action = "RIGHT"

5. DRIFTING (near a white edge or seeing green):
   - Near LEFT edge: Action = "RIGHT"
   - Near RIGHT edge: Action = "LEFT"

6. OSCILLATION: If your last 3 actions alternated LEFT and RIGHT, choose FORWARD or steer toward the yellow line.

STEP 4 — RETURN JSON only, no extra text:
{
  "left_column": "what you see",
  "center_column": "what you see",
  "right_column": "what you see",
  "yellow_line_top": "drifts LEFT | drifts RIGHT | centered",
  "road_curve": "LEFT|RIGHT|STRAIGHT",
  "action": "FORWARD|LEFT|RIGHT|STOP",
  "reasoning": "one sentence explanation"
}
"""

# Compact prompt for local models
SYSTEM_PROMPT_LOCAL = """You are a robot navigation controller. The robot is ~2.55 m long, ~1.27 m wide, driving on a gray asphalt road.

ROAD MARKINGS:
- YELLOW center line: use this to detect road direction and stay centered.
- WHITE edge lines: road boundaries — never cross them.
- Green surface: off-road, avoid.

PROXIMITY RULE — use apparent size to judge if an obstacle is a threat:
- If an obstacle appears in the BOTTOM THIRD of the image → it is CLOSE → react to it.
- If an obstacle appears only in the TOP HALF of the image → it is FAR AWAY → ignore it, keep going FORWARD.
- A small or distant-looking object near the top is NOT a reason to stop or turn.

ANALYZE IN ORDER:

1. COLUMNS — describe what you see in LEFT / CENTER / RIGHT thirds of the image:
   Use only: asphalt | cone_close | cone_far | white_line | yellow_line | green | wall | unclear
   cone_close = cone visible in bottom third of image
   cone_far   = cone visible only in top half of image (ignore it)

2. YELLOW LINE DIRECTION — trace the yellow center line from bottom to top:
   - Drifts left in top half → road curves LEFT
   - Drifts right in top half → road curves RIGHT
   - Stays centered → STRAIGHT

3. ACTION — pick exactly one using this priority:
   - STOP: all three columns have cone_close or wall at bottom. Absolute last resort.
   - LEFT: road curves left, OR cone_close in center column with left side clear
   - RIGHT: road curves right, OR cone_close in center column with right side clear
   - FORWARD: anything else — default to this. cone_far is never a reason to stop or turn.
   - If near white edge: steer away from it.

OUTPUT — respond with ONLY this JSON, no extra text:
{
  "left_column": "<what you see>",
  "center_column": "<what you see>",
  "right_column": "<what you see>",
  "yellow_line": "drifts_left | drifts_right | centered",
  "action": "FORWARD | LEFT | RIGHT | STOP",
  "reasoning": "<max 8 words>"
}"""

ACTIVE_PROMPT  = SYSTEM_PROMPT_OPENAI if LLM_BACKEND == "openai" else SYSTEM_PROMPT_LOCAL
COL_LEFT_KEY   = "left_column"
COL_CENTER_KEY = "center_column"
COL_RIGHT_KEY  = "right_column"
REASONING_KEY  = "reasoning"

LLM_INTERVAL = float(os.getenv("LLM_INTERVAL", "1.5"))

_img_w, _img_h = os.getenv("IMAGE_SIZE", "320x240").split("x")
IMAGE_WIDTH  = int(_img_w)
IMAGE_HEIGHT = int(_img_h)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))


class LLMDriverNode(Node):
    def __init__(self):
        super().__init__('llm_driver_node')

        self.client = client
        self.model  = MODEL
        self.bridge = CvBridge()
        self.latest_image = None

        self.history     = []
        self.max_history = 5

        self.image_sub  = self.create_subscription(
            Image, '/car_camera/image_raw', self.img_cb, 10)
        self.intent_pub = self.create_publisher(AgentIntent, '/agent/intent', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(LLM_INTERVAL, self.process_llm_logic)
        self.get_logger().info(
            f"LLM Driver Node ready. "
            f"Backend: {LLM_BACKEND.upper()} | Model: {self.model} | "
            f"Interval: {LLM_INTERVAL}s | Image: {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ Q{JPEG_QUALITY}"
        )

    def img_cb(self, msg):
        self.latest_image = msg

    def _extract_json(self, raw: str) -> dict:
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")

        # Strip <think>...</think> blocks for thinking models
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        # Strip markdown code fences
        raw = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw, flags=re.DOTALL).strip()

        # Try to find and parse a JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Partial recovery: pull action field even from truncated response
        action_match = re.search(r'"action"\s*:\s*"(FORWARD|LEFT|RIGHT|STOP)"', raw)
        if action_match:
            reason_match = re.search(
                rf'"{REASONING_KEY}"\s*:\s*"([^"]*)', raw
            )
            return {
                "action":       action_match.group(1),
                REASONING_KEY:  reason_match.group(1) if reason_match else "truncated",
            }

        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    def process_llm_logic(self):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet, skipping.")
            return

        # Encode image
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        resized = cv2.resize(cv_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, buffer = cv2.imencode('.jpg', resized, encode_params)
        base64_img = base64.b64encode(buffer).decode('utf-8')

        # Build messages
        messages = [{"role": "system", "content": ACTIVE_PROMPT}]

        if self.history:
            history_str = " -> ".join([h['action'] for h in self.history])
            messages.append({
                "role":    "user",
                "content": f"My last {len(self.history)} actions were: {history_str}.",
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                {"type": "text",
                 "text": "This is my current view. What is my next action?"},
            ],
        })

        try:
            # NOTE: no extra_body — Qwen2.5-VL does not support {"think": False}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=256,
            )

            raw_content = response.choices[0].message.content or ""
            self.get_logger().debug(f"Raw LLM response: {raw_content[:300]}")
            res = self._extract_json(raw_content)

            action    = res.get('action', 'STOP')
            reasoning = res.get(REASONING_KEY, '')

            self.history.append({"action": action, REASONING_KEY: reasoning})
            if len(self.history) > self.max_history:
                self.history.pop(0)

            if action not in ["FORWARD", "LEFT", "RIGHT", "STOP"]:
                self.get_logger().warn(f"Invalid action: {action} | full response: {res}")
                self.publish_stop()
                return

            self.publish_commands(action, reasoning, res)

        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            self.publish_stop()

    def publish_commands(self, action: str, reasoning: str, res: dict):
        intent = AgentIntent()
        intent.intent    = action
        intent.reasoning = reasoning
        self.intent_pub.publish(intent)

        msg = Twist()
        if LLM_BACKEND == "local":
            # Qwen2.5-VL on AMD server
            if action == "FORWARD":
                msg.linear.x  = 0.2
                msg.angular.z = 0.0
            elif action == "LEFT":
                msg.linear.x  = 0.1
                msg.angular.z = 0.05
            elif action == "RIGHT":
                msg.linear.x  = 0.1
                msg.angular.z = -0.05
            elif action == "STOP":
                msg.linear.x  = 0.0
                msg.angular.z = 0.0
        else:
            # OpenAI
            if action == "FORWARD":
                msg.linear.x  = 0.4
                msg.angular.z = 0.0
            elif action == "LEFT":
                msg.linear.x  = 0.2
                msg.angular.z = 0.1
            elif action == "RIGHT":
                msg.linear.x  = 0.2
                msg.angular.z = -0.1
            elif action == "STOP":
                msg.linear.x  = 0.0
                msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)

        self.get_logger().info(
            f"ACTION: {action} | "
            f"L: {res.get(COL_LEFT_KEY, '?')} | "
            f"C: {res.get(COL_CENTER_KEY, '?')} | "
            f"R: {res.get(COL_RIGHT_KEY, '?')} | "
            f"REASON: {reasoning} | "
            f"HISTORY: {[h['action'] for h in self.history]}"
        )

    def publish_stop(self):
        msg = Twist()
        msg.linear.x  = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().warn("Fallback: published STOP due to LLM error.")


def main():
    rclpy.init()
    node = LLMDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()