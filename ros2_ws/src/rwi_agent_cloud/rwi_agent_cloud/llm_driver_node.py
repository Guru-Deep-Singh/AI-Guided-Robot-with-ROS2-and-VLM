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
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the current working directory / parents (default behavior)
load_dotenv()

# Fallback: explicitly load .env from the project root if keys are still missing
if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_BACKEND"):
    project_root = Path(__file__).resolve().parents[4]
    load_dotenv(project_root / ".env")

# --- Backend config ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  # "openai" or "ollama"

if LLM_BACKEND == "ollama":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL must be set in .env when using LLM_BACKEND=ollama")
    MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY
    )
else:
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
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

SYSTEM_PROMPT_ollama = """
You are an autonomous driving system controlling a robot approximately 2.55 m long and 1.27 m wide.

Your goal is to move forward while staying on the gray asphalt road. If a obstacle is detected close to the robot, avoid it and continue moving forward. However, if it is far from the robot, ignore it and continue moving forward.

ROAD LAYOUT:
- The road has a YELLOW center line and WHITE edge lines on both sides.
- You must NEVER cross the WHITE edge lines.
- You MAY cross the YELLOW center line only to avoid an obstacle.
- Green surface is NOT drivable.

RETURN JSON only, no extra text:
{
  "action": "FORWARD|LEFT|RIGHT|STOP",
  "reasoning": "one sentence explanation"
}
"""

LLM_INTERVAL = float(os.getenv("LLM_INTERVAL", "1.5"))

# Image encoding settings — smaller = faster LLM response, lower quality
_img_w, _img_h = os.getenv("IMAGE_SIZE", "160x120").split("x")
IMAGE_WIDTH = int(_img_w)
IMAGE_HEIGHT = int(_img_h)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "50"))  # 0-100, lower = smaller payload

class LLMDriverNode(Node):
    def __init__(self):
        super().__init__('llm_driver_node')

        self.client = client
        self.model = MODEL
        self.bridge = CvBridge()
        self.latest_image = None

        # Sliding window memory of recent decisions
        self.history = []
        self.max_history = 5

        self.image_sub = self.create_subscription(
            Image, '/car_camera/image_raw', self.img_cb, 10)
        self.intent_pub = self.create_publisher(AgentIntent, '/agent/intent', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(LLM_INTERVAL, self.process_llm_logic)
        self.get_logger().info(
            f"LLM Driver Node ready. Backend: {LLM_BACKEND.upper()} | Model: {self.model} | "
            f"Interval: {LLM_INTERVAL}s | Image: {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ Q{JPEG_QUALITY}"
        )

    def img_cb(self, msg):
        self.latest_image = msg

    def process_llm_logic(self):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet, skipping.")
            return

        # Encode image
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        _, buffer = cv2.imencode('.jpg', cv2.resize(cv_img, (IMAGE_WIDTH, IMAGE_HEIGHT))) #[cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        base64_img = base64.b64encode(buffer).decode('utf-8')

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.history:
            history_str = " -> ".join([h['action'] for h in self.history])
            messages.append({
                "role": "user",
                "content": f"My last {len(self.history)} actions were: {history_str}."
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "This is my current view. What is my next action? Do not think. Answer immediately."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                #response_format={"type": "json_object"},
                max_tokens=200
            )

            res = json.loads(response.choices[0].message.content)
            action = res['action']
            reasoning = res['reasoning']

            self.history.append({"action": action, "reasoning": reasoning})
            if len(self.history) > self.max_history:
                self.history.pop(0)

            self.publish_commands(action, reasoning, res)

        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            self.publish_stop()

    def publish_commands(self, action, reasoning, res):
        intent = AgentIntent()
        intent.intent = action
        intent.reasoning = reasoning
        self.intent_pub.publish(intent)

        msg = Twist()
        if LLM_BACKEND == "ollama":
            if action == "FORWARD":
                msg.linear.x = 0.3
                msg.angular.z = 0.0
            elif action == "LEFT":
                msg.linear.x = 0.1
                msg.angular.z = 0.05
            elif action == "RIGHT":
                msg.linear.x = 0.1
                msg.angular.z = -0.05
            elif action == "STOP":
                msg.linear.x = 0.0
                msg.angular.z = 0.0
        else:
            if action == "FORWARD":
                msg.linear.x = 0.4
                msg.angular.z = 0.0
            elif action == "LEFT":
                msg.linear.x = 0.2
                msg.angular.z = 0.1
            elif action == "RIGHT":
                msg.linear.x = 0.2
                msg.angular.z = -0.1
            elif action == "STOP":
                msg.linear.x = 0.0
                msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)

        self.get_logger().info(
            f"ACTION: {action} | "
            f"L: {res.get('left_column','?')} | "
            f"C: {res.get('center_column','?')} | "
            f"R: {res.get('right_column','?')} | "
            f"REASON: {reasoning} | "
            f"HISTORY: {[h['action'] for h in self.history]}"
        )

    def publish_stop(self):
        msg = Twist()
        msg.linear.x = 0.0
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