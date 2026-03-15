import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from rwi_interfaces.msg import AgentIntent
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import base64
import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_BACKEND"):
    load_dotenv(os.path.join(os.path.expanduser('~'), 'git_reps/AI-Guided-Robot-with-ROS2-and-VLM', '.env'))

# ------- Backend config -------
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  # "openai" or "local"

if LLM_BACKEND == "local":
    LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "").rstrip("/")
    if LOCAL_BASE_URL.endswith("/chat/completions"):
        LOCAL_BASE_URL = LOCAL_BASE_URL[: -len("/chat/completions")]
    if not LOCAL_BASE_URL:
        raise ValueError("LOCAL_BASE_URL must be set in .env when using LLM_BACKEND=local")
    LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "none")
    MODEL  = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    client = OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)
else:
    MODEL  = os.getenv("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LOCAL_MODEL_TYPE switches prompt and response parsing when using a local backend.
#   "general"    — use SYSTEM_PROMPT_LOCAL, expects full JSON with action/reasoning
#   "finetuned"  — use SYSTEM_PROMPT_LOCAL_FINE_TUNED, expects {"intent": "X"} only
LOCAL_MODEL_TYPE = os.getenv("LOCAL_MODEL_TYPE", "general").lower()

# ------- Prompts -------

SYSTEM_PROMPT_OPENAI = """
You are an autonomous driving system controlling a robot approximately 2.55 m long and 1.27 m wide.

Your goal is to move forward while staying on the gray asphalt road between the white road lines. If an obstacle is detected close to the robot, avoid it and continue moving forward. However, if it is far from the robot, ignore it and continue moving forward.
You must STAY ON THE ROAD. Do not drive on the grass or off-road. You must NOT cross the WHITE ROAD LINES.
Being on center of the road is ideal but not mandatory. You can be anywhere between the white lines.

ROAD LAYOUT:
- The road has a YELLOW center line and WHITE edge lines on both sides.
- You must NEVER cross the WHITE edge lines. Always stay between the left and right white lines which has yellow center line.
- You MAY cross the YELLOW center line only to avoid an obstacle.
- Green surface is NOT drivable.

YOU WILL RECEIVE TWO IMAGES:
1. CAMERA IMAGE — forward-facing RGB view from the robot.
2. LIDAR MAP — a top-down radar-style image of the robot's surroundings.
   - The robot is at the CENTER of the LiDAR image (cyan dot), facing UP.
   - WHITE dots are detected obstacles/walls.
   - The colored rings show distance: innermost = 4m, middle = 8m, outer = 12m.
   - LEFT in the LiDAR image = robot's left. RIGHT = robot's right.
   - Use this to judge how close obstacles really are and whether a gap exists.

STEP 1 — ANALYZE THE CAMERA IMAGE:
Divide the image into three vertical columns: LEFT third, CENTER third, RIGHT third.
Divide vertically into TOP half (far ahead) and BOTTOM half (close to robot).
For each column report: asphalt, cone, white line, yellow line, green surface, or wall.

STEP 2 — DETECT ROAD DIRECTION using ONLY the YELLOW center line:
- Trace the YELLOW center line from the BOTTOM of the image upward to the TOP.
- If the yellow line drifts LEFT in the TOP half → road curves LEFT → turn LEFT.
- If the yellow line drifts RIGHT in the TOP half → road curves RIGHT → turn RIGHT.
- If the yellow line stays centered → road is STRAIGHT → go FORWARD.
- Do NOT use white edge lines to determine curve direction.

STEP 3 — CHECK LIDAR MAP for nearby obstacles:
- The LiDAR image has labeled zones: AHEAD (top), LEFT (left side), RIGHT (right side).
- WHITE dots in LEFT or RIGHT zones = road boundary walls. IGNORE them — always present on a road.
- WHITE dots in the AHEAD zone within 4m = real obstacle in your path → react.
- WHITE dots in the AHEAD zone beyond 4m → far, ignore and keep FORWARD.
- NEVER steer because of dots in the LEFT or RIGHT zones — those are road edges not obstacles.
- Use LiDAR AHEAD zone only to confirm camera obstacles and determine which side has a gap.

STEP 4 — DECIDE ACTION using this priority order:

1. STOP: Only if ALL directions are blocked. Absolute last resort.

2. REVERSE: Use when completely off-road with no lines visible, OR stuck (last 3+ actions same with no improvement).
   Do NOT use REVERSE just because there is a cone ahead — check LiDAR first.

3. ROAD CURVES (Step 2): Curve LEFT → "LEFT". Curve RIGHT → "RIGHT".

4. FORWARD: Yellow line centered, CENTER column clear, not near white edge.

5. OBSTACLE (cone in CENTER column): LiDAR gap LEFT → "LEFT". LiDAR gap RIGHT → "RIGHT".

6. DRIFTING (near white edge or seeing green): Near LEFT edge → "RIGHT". Near RIGHT edge → "LEFT".

7. OSCILLATION: Last 3 actions alternated LEFT/RIGHT → choose FORWARD or steer toward yellow line.

STEP 5 — RETURN JSON only, no extra text:
{
  "left_column": "what you see",
  "center_column": "what you see",
  "right_column": "what you see",
  "yellow_line_top": "drifts LEFT | drifts RIGHT | centered | not visible",
  "road_curve": "LEFT|RIGHT|STRAIGHT|UNKNOWN",
  "lidar_summary": "one sentence about what the LiDAR map shows",
  "action": "FORWARD|LEFT|RIGHT|REVERSE|STOP",
  "reasoning": "one sentence explanation"
}
"""

# Compact prompt for general local models like Qwen2.5-VL-72B
SYSTEM_PROMPT_LOCAL = """You are a robot navigation controller. Robot is ~2.55m long, ~1.27m wide, on a gray asphalt road.
Your task is to drive the robot on the road. Always try to keep the robot in the center of the road, close to the yellow line.
If you can not see the yellow line, Always try to keep the robot in the center of the road i.e. more asphalt area.

ROAD MARKINGS:
- YELLOW center line: follow it to stay centered and detect road curves.
- WHITE edge lines: hard boundaries — never cross them.
- Green surface: off-road, avoid.

PROXIMITY RULE (use image position as depth proxy):
- Obstacle in BOTTOM THIRD of camera image → CLOSE → react.
- Obstacle only in TOP HALF of camera image → FAR → ignore, keep going FORWARD.

YOU WILL RECEIVE TWO IMAGES:
1. CAMERA — forward RGB view.
2. LIDAR MAP — top-down view. Robot = cyan dot at center, facing UP.
   - WHITE dots = obstacles. Rings = 4m / 8m / 12m distance.
   - Use LiDAR to confirm if a camera obstacle is truly close, and which side has more gap.

ANALYZE IN ORDER:

1. COLUMNS — LEFT / CENTER / RIGHT thirds of camera image:
   Use only: asphalt | cone_close | cone_far | white_line | yellow_line | green | wall | unclear
   cone_close = cone in bottom third. cone_far = cone only in top half (ignore it).

2. YELLOW LINE — trace from bottom to top:
   - Drifts left in top half → LEFT curve
   - Drifts right in top half → RIGHT curve
   - Stays centered → STRAIGHT
   ALWAYS use yellow line to determine the road direction. If you can not see yellow line, use white lines to determine the road direction. Give them priority over the distance of obstacle during curves.
   Follow the Line even if it means to go close to the obstacle.

3. LIDAR — read the labeled zones (AHEAD / LEFT / RIGHT):
   - WHITE dots in the LEFT or RIGHT zones = road boundary walls. IGNORE them — they are always there.
   - WHITE dots in the AHEAD zone within 4m = real obstacle directly in your path → react.
   - WHITE dots in the AHEAD zone beyond 4m = far obstacle → ignore, keep FORWARD.
   - ONLY react to dots in the AHEAD / center zone. Never steer because of LEFT or RIGHT zone dots.

4. ACTION — pick exactly one:
   - STOP: AHEAD blocked AND camera shows all columns blocked. Absolute last resort.
   - REVERSE: completely off-road with no lines visible, OR stuck 5+ steps same direction. Use only when you can not see any road or yellow line. If you see road steer towards it using LEFT or RIGHT.
   - LEFT: road curves left, OR cone_close in camera center AND LiDAR AHEAD gap is on left.
   - RIGHT: road curves right, OR cone_close in camera center AND LiDAR AHEAD gap is on right.
   - FORWARD: default — use this unless above conditions clearly apply.
     Dots in LEFT/RIGHT LiDAR zones are road walls — NEVER steer away from them.

OUTPUT — ONLY this JSON, no extra text:
{
  "left_column": "<what you see>",
  "center_column": "<what you see>",
  "right_column": "<what you see>",
  "yellow_line": "drifts_left | drifts_right | centered | not_visible",
  "lidar_summary": "<one sentence>",
  "action": "FORWARD | LEFT | RIGHT | REVERSE | STOP",
  "reasoning": "<max 8 words>"
}"""

# Minimal prompt for the fine-tuned 7B model — matches training exactly.
# The model has learned driving behaviour in its weights; it does not need
# step-by-step instructions, only the same prompt it saw during fine-tuning.
SYSTEM_PROMPT_LOCAL_FINE_TUNED = (
    "You are a robot navigation controller. "
    "Given a forward camera image and a top-down LiDAR map, "
    "output only a JSON object with a single key 'intent' "
    "with value one of: FORWARD, LEFT, RIGHT, REVERSE, STOP."
)

USER_PROMPT_LOCAL_FINE_TUNED = (
    "IMAGE 1: forward camera view. "
    "IMAGE 2: top-down LiDAR map (robot = cyan dot at centre facing up, "
    "rings = 4m/8m/12m, white dots = obstacles). "
    "What is the next driving action?"
)

# ------- Active prompt selection -------
if LLM_BACKEND == "openai":
    ACTIVE_PROMPT = SYSTEM_PROMPT_OPENAI
elif LOCAL_MODEL_TYPE == "finetuned":
    ACTIVE_PROMPT = SYSTEM_PROMPT_LOCAL_FINE_TUNED
else:
    ACTIVE_PROMPT = SYSTEM_PROMPT_LOCAL

# ------- Runtime config -------
LLM_INTERVAL = float(os.getenv("LLM_INTERVAL", "1.5"))

_img_w, _img_h = os.getenv("IMAGE_SIZE", "320x320").split("x") # Camera is 800x800 square — keep square to avoid aspect ratio distortion
IMAGE_WIDTH  = int(_img_w)
IMAGE_HEIGHT = int(_img_h)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

LIDAR_IMG_SIZE  = 256
LIDAR_MAX_RANGE = 12.0
LIDAR_FOV_DEG   = 120.0   # actual sensor FOV from URDF (−60° to +60°)
LIDAR_RINGS_M   = [4.0, 8.0, 12.0]


class LLMDriverNode(Node):
    def __init__(self):
        super().__init__('llm_driver_node')

        self.client = client
        self.model  = MODEL
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_scan  = None

        self.history     = []
        self.max_history = 5

        self.image_sub = self.create_subscription(
            Image,     '/car_camera/image_raw', self.img_cb,  10)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan',                 self.scan_cb, 10)

        self.intent_pub    = self.create_publisher(AgentIntent, '/agent/intent',    10)
        self.cmd_vel_pub   = self.create_publisher(Twist,       '/cmd_vel',         10)
        self.lidar_img_pub = self.create_publisher(Image,       '/llm/lidar_image', 10)

        self.timer = self.create_timer(LLM_INTERVAL, self.process_llm_logic)
        self.get_logger().info(
            f"LLM Driver Node ready. "
            f"Backend: {LLM_BACKEND.upper()} | "
            f"Model type: {LOCAL_MODEL_TYPE if LLM_BACKEND == 'local' else 'openai'} | "
            f"Model: {self.model} | "
            f"Interval: {LLM_INTERVAL}s | Image: {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ Q{JPEG_QUALITY}"
        )

    # ------- Sensor callbacks -------

    def img_cb(self, msg):
        self.latest_image = msg

    def scan_cb(self, msg):
        self.latest_scan = msg
        try:
            lidar_bgr = self.render_lidar_bgr()
            if lidar_bgr is not None:
                ros_img = self.bridge.cv2_to_imgmsg(lidar_bgr, encoding='bgr8')
                ros_img.header.stamp    = msg.header.stamp
                ros_img.header.frame_id = 'lidar_link'
                self.lidar_img_pub.publish(ros_img)
            else:
                self.get_logger().warn("render_lidar_bgr returned None", throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f"LiDAR image render/publish failed: {e}")

    # ------- LiDAR: top-down image -------

    def render_lidar_bgr(self):
        """Render 2D LiDAR (120° FOV, −60° to +60°) as a top-down BGR image.

        Robot = cyan dot at centre, facing UP.
        White dots = obstacles. Coloured rings = 4 / 8 / 12 m.
        Gray wedge = the 120° sensor coverage cone.
        Dark region outside the wedge = blind zone (no sensor data).
        """
        if self.latest_scan is None:
            return None

        scan   = self.latest_scan
        size   = LIDAR_IMG_SIZE
        cx, cy = size // 2, size // 2
        scale  = (size // 2 - 4) / LIDAR_MAX_RANGE

        img = np.zeros((size, size, 3), dtype=np.uint8)

        # ------- Draw 120° coverage wedge so the LLM can see the sensor FOV ------- 
        # The sensor sweeps −60° to +60° around forward (up in image).
        # In image coords: forward=up, left=left. Angles from +y axis (up).
        # cv2.ellipse angles are measured from +x axis (right), clockwise.
        # sensor left edge  = −60° from forward = 30° from +x axis  (image: 210°)
        # sensor right edge = +60° from forward = 150° from +x axis (image: 330°... wait)
        # Simpler: draw filled polygon for the wedge.
        half_fov = np.radians(LIDAR_FOV_DEG / 2.0)   # 60°
        r_px = int(LIDAR_MAX_RANGE * scale)
        n_pts = 32
        wedge_angles = np.linspace(-half_fov, half_fov, n_pts)
        # In image: forward=up → col offset = sin(a), row offset = -cos(a)
        wx = (cx + r_px * np.sin(wedge_angles)).astype(int)
        wy = (cy - r_px * np.cos(wedge_angles)).astype(int)
        wedge_pts = np.column_stack([wx, wy])
        wedge_pts = np.vstack([[cx, cy], wedge_pts, [cx, cy]])
        cv2.fillPoly(img, [wedge_pts.astype(np.int32)], (30, 30, 30))

        # ------- Distance rings ------- 
        ring_colors = [(100, 60, 60), (60, 100, 60), (60, 60, 100)]
        for dist, color in zip(LIDAR_RINGS_M, ring_colors):
            cv2.circle(img, (cx, cy), int(dist * scale), color, 1)

        # ------- Forward direction tick ------- 
        cv2.line(img, (cx, cy), (cx, cy - r_px), (80, 80, 80), 1)

        # ------- FOV boundary lines (show edge of 120° cone) ------- 
        for sign in [-1, 1]:
            ex = int(cx + r_px * np.sin(sign * half_fov))
            ey = int(cy - r_px * np.cos(sign * half_fov))
            cv2.line(img, (cx, cy), (ex, ey), (60, 60, 60), 1)

        # ------- Zone labels so the LLM knows spatial layout ------- 
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        font_color = (180, 180, 180)
        thickness  = 1
        # AHEAD label — top centre
        cv2.putText(img, "AHEAD", (cx - 22, 12), font, font_scale, font_color, thickness)
        # LEFT label — middle left
        cv2.putText(img, "LEFT",  (4, cy),        font, font_scale, font_color, thickness)
        # RIGHT label — middle right
        cv2.putText(img, "RIGHT", (size - 38, cy),font, font_scale, font_color, thickness)
        # NEAR label — just above robot
        cv2.putText(img, "4m",  (cx + 6, cy - int(4.0  * scale)), font, font_scale, (80,80,80), thickness)
        cv2.putText(img, "8m",  (cx + 6, cy - int(8.0  * scale)), font, font_scale, (80,80,80), thickness)
        cv2.putText(img, "12m", (cx + 6, cy - int(11.5 * scale)), font, font_scale, (80,80,80), thickness)

        # ------- Robot marker (cyan dot) ------- 
        cv2.circle(img, (cx, cy), 4, (0, 200, 255), -1)

        # ------- Scan points ------- 
        angles = (scan.angle_min
                  + np.arange(len(scan.ranges)) * scan.angle_increment)
        ranges = np.array(scan.ranges, dtype=np.float32)

        valid  = (np.isfinite(ranges)
                  & (ranges > scan.range_min)
                  & (ranges < LIDAR_MAX_RANGE))
        angles = angles[valid]
        ranges = ranges[valid]

        # ROS REP-103: angle 0 = forward, positive CCW = left
        px_col = (cx - ranges * np.sin(angles) * scale).astype(int)
        px_row = (cy - ranges * np.cos(angles) * scale).astype(int)

        in_bounds = ((px_col >= 0) & (px_col < size)
                     & (px_row >= 0) & (px_row < size))
        for c, r in zip(px_col[in_bounds], px_row[in_bounds]):
            cv2.circle(img, (int(c), int(r)), 2, (255, 255, 255), -1)

        return img

    def render_lidar_base64(self):
        bgr = self.render_lidar_bgr()
        if bgr is None:
            return None
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    # ------- JSON extraction ------- 
    # This is critial fix for local models as sometimes they can not comply withn JSON 
    # format and return extra text or markdown. This function extracts the JSON 
    # and returns it as a dictionary.
    def _extract_json(self, raw: str) -> dict:
        if not raw or not raw.strip():
            raise ValueError("Empty response from LLM")

        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw, flags=re.DOTALL).strip()

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract intent (fine-tuned model) or action (general model)
        for key in ("intent", "action"):
            key_match = re.search(
                rf'"{key}"\s*:\s*"(FORWARD|LEFT|RIGHT|REVERSE|STOP)"', raw
            )
            if key_match:
                return {key: key_match.group(1)}

        raise ValueError(f"No JSON found in response: {raw[:200]}")

    # ------- Main LLM loop -------

    def process_llm_logic(self):
        if self.latest_image is None:
            self.get_logger().warn("Waiting for camera image...")
            return
        if self.latest_scan is None:
            self.get_logger().warn("Waiting for LiDAR scan...")
            return

        cv_img  = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        resized = cv2.resize(cv_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        _, buf  = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        cam_b64 = base64.b64encode(buf).decode('utf-8')

        lidar_b64 = self.render_lidar_base64()
        
        # ── Build messages depending on model type ────────────────────────────
        messages = [{"role": "system", "content": ACTIVE_PROMPT}]

        if LOCAL_MODEL_TYPE == "finetuned" and LLM_BACKEND == "local":
            # Fine-tuned model: no history injection, exact training prompt
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lidar_b64}"}},
                {"type": "text",      "text": USER_PROMPT_LOCAL_FINE_TUNED},
            ]
            messages.append({"role": "user", "content": content})

        else:
            # General model (72B or OpenAI): history context + descriptive text
            if self.history:
                history_str = " -> ".join([h['action'] for h in self.history])
                messages.append({
                    "role":    "user",
                    "content": f"My last {len(self.history)} actions were: {history_str}.",
                })

            if lidar_b64:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
                    {"type": "text",      "text": "IMAGE 1 above: Camera (forward view). IMAGE 2 below: LiDAR map (top-down, robot = cyan dot at center facing up, rings = 4m/8m/12m)."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lidar_b64}"}},
                    {"type": "text",      "text": "What is my next action?"},
                ]
            else:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
                    {"type": "text",      "text": "Camera view only (LiDAR unavailable). What is my next action?"},
                ]
            messages.append({"role": "user", "content": content})

        # ── Call LLM ──────────────────────────────────────────────────────────
        try:
            if LLM_BACKEND == "local":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=64 if LOCAL_MODEL_TYPE == "finetuned" else 256,
                    temperature=0.1,
                )
                raw_content = response.choices[0].message.content or ""
                self.get_logger().debug(f"Raw LLM response: {raw_content[:300]}")
                res = self._extract_json(raw_content)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                res = json.loads(response.choices[0].message.content)

            # ── Resolve action — fine-tuned uses "intent", general uses "action"
            action = res.get('intent') or res.get('action', 'STOP')
            reasoning = res.get('reasoning', '')

            if action not in ["FORWARD", "LEFT", "RIGHT", "REVERSE", "STOP"]:
                self.get_logger().warn(f"Invalid action '{action}' — stopping.")
                self.publish_stop()
                return

            self.history.append({"action": action, "reasoning": reasoning})
            if len(self.history) > self.max_history:
                self.history.pop(0)

            self.publish_commands(action, reasoning, res)

        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            self.publish_stop()

    # ------- Command publishing -------

    def publish_commands(self, action: str, reasoning: str, res: dict):
        intent           = AgentIntent()
        intent.intent    = action
        intent.reasoning = reasoning
        self.intent_pub.publish(intent)

        msg = Twist()
        if LLM_BACKEND == "local":
            if action == "FORWARD":
                msg.linear.x  =  0.3;  msg.angular.z =  0.0
            elif action == "LEFT":
                msg.linear.x  =  0.2;  msg.angular.z =  0.05
            elif action == "RIGHT":
                msg.linear.x  =  0.2;  msg.angular.z = -0.05
            elif action == "REVERSE":
                msg.linear.x  = -0.2;  msg.angular.z =  0.0
            elif action == "STOP":
                msg.linear.x  =  0.0;  msg.angular.z =  0.0
        else:
            if action == "FORWARD":
                msg.linear.x  =  0.4;  msg.angular.z =  0.0
            elif action == "LEFT":
                msg.linear.x  =  0.2;  msg.angular.z =  0.1
            elif action == "RIGHT":
                msg.linear.x  =  0.2;  msg.angular.z = -0.1
            elif action == "REVERSE":
                msg.linear.x  = -0.3;  msg.angular.z =  0.0
            elif action == "STOP":
                msg.linear.x  =  0.0;  msg.angular.z =  0.0

        self.cmd_vel_pub.publish(msg)

        # Log format adapts to model type
        if LOCAL_MODEL_TYPE == "finetuned" and LLM_BACKEND == "local":
            self.get_logger().info(
                f"ACTION: {action} | "
                f"HISTORY: {[h['action'] for h in self.history]}"
            )
        else:
            self.get_logger().info(
                f"ACTION: {action} | "
                f"L: {res.get('left_column','?')} | "
                f"C: {res.get('center_column','?')} | "
                f"R: {res.get('right_column','?')} | "
                f"LIDAR: {res.get('lidar_summary','?')} | "
                f"REASON: {reasoning} | "
                f"HISTORY: {[h['action'] for h in self.history]}"
            )

    def publish_stop(self):
        self.cmd_vel_pub.publish(Twist())
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