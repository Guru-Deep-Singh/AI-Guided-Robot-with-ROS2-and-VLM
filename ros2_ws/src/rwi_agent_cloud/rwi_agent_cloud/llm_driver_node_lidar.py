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
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(os.path.join(os.path.expanduser('~'), 'ros-with-ai', '.env'))

SYSTEM_PROMPT = """
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
   - The robot is at the CENTER of the LiDAR image (yellow dot), facing UP. The WHITE thin line is the direction of motion of the robot.
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
- In the BOTTOM half, the yellow line is roughly centered. Note where it goes in the TOP half.
- If the yellow line drifts to the LEFT in the TOP half: the road curves LEFT → you must turn LEFT.
- If the yellow line drifts to the RIGHT in the TOP half: the road curves RIGHT → you must turn RIGHT.
- If the yellow line stays centered: road is STRAIGHT → go FORWARD.
- IMPORTANT: Do NOT use the white edge lines to determine curve direction. White lines are edges and can be misleading. Only use the YELLOW center line.

STEP 3 — CHECK LIDAR MAP for nearby obstacles:
- If the WHITE dot does not appear in the LIDAR MAP in the direction of motion → treat as no obstacle.
- If WHITE dots appear inside the innermost ring (within 4m) directly ahead of the yellow dot on LIDAR MAP in the direction of motion → treat as immediate hazard.
- If WHITE dots appear inside the middle ring (within 8m) directly ahead of the yellow dot on LIDAR MAP in the direction of motion → treat as mid-range obstacle. 
- If WHITE dots appear inside the outer ring (within 12m) directly ahead of the yellow dot on LIDAR MAP in the direction of motion → treat as far-range obstacle.
- Use the LiDAR to confirm whether a visible object in the camera is a real obstacle.
- Use the LiDAR to identify which side (left/right) has a larger gap to steer into.

STEP 4 — DECIDE ACTION using this priority order:

1. STOP: Only if ALL directions are blocked AND reversing would not help. Absolute last resort.

2. REVERSE: Use when:
   - You are completely off-road and can not see any yellow or white lines.
   - You are blocked completely by obstacles in all three columns of the RGB image.
   - Your last 3+ actions were all LEFT or all RIGHT with no improvement (stuck).
   - Do NOT use REVERSE just because there is a cone ahead. CHECK LIDAR MAP for obstacles first.

3. ROAD CURVES (from Step 2, yellow line analysis):
   - Curve LEFT: Action = "LEFT"
   - Curve RIGHT: Action = "RIGHT"
   - Only apply this if you are confident the yellow line is clearly visible and curving.

4. FORWARD: If the yellow line is centered and straight, CENTER column is clear, and you are not near a white edge.

5. OBSTACLE (cone in CENTER column):
   - LiDAR shows larger gap on LEFT → "LEFT"
   - LiDAR shows larger gap on RIGHT → "RIGHT"

6. DRIFTING (near a white edge or seeing green):
   - Near LEFT edge: Action = "RIGHT"
   - Near RIGHT edge: Action = "LEFT"

7. OSCILLATION: If last 3 actions alternated LEFT/RIGHT, choose FORWARD or steer toward yellow line.

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

LLM_INTERVAL = 1.5  # seconds between LLM decisions

# LiDAR rendering parameters
LIDAR_IMG_SIZE  = 256   # pixels square
LIDAR_MAX_RANGE = 12.0  # metres — anything beyond this is ignored
LIDAR_RINGS_M   = [4.0, 8.0, 12.0]   # distance rings to draw


class LLMDriverNode(Node):
    def __init__(self):
        super().__init__('llm_driver_node')

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_scan  = None

        # Sliding window memory of recent decisions
        self.history = []
        self.max_history = 5

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,     '/car_camera/image_raw', self.img_cb,  10)
        self.scan_sub  = self.create_subscription(
            LaserScan, '/scan',                 self.scan_cb, 10)

        self.intent_pub    = self.create_publisher(AgentIntent, '/agent/intent',    10)
        self.cmd_vel_pub   = self.create_publisher(Twist,       '/cmd_vel',         10)
        # Publishes the LiDAR bird's-eye image so it can be viewed in RViz
        self.lidar_img_pub = self.create_publisher(Image,       '/llm/lidar_image', 10)

        self.timer = self.create_timer(LLM_INTERVAL, self.process_llm_logic)
        self.get_logger().info(f"LLM Driver Node ready. Decision interval: {LLM_INTERVAL}s")

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def img_cb(self, msg):
        self.latest_image = msg

    def scan_cb(self, msg):
        self.latest_scan = msg
        # Publish the LiDAR image every time a new scan arrives (~10 Hz),
        # not just on LLM ticks, so RViz stays fluid.
        lidar_bgr = self.render_lidar_bgr()
        if lidar_bgr is not None:
            ros_img = self.bridge.cv2_to_imgmsg(lidar_bgr, encoding='bgr8')
            ros_img.header.stamp = msg.header.stamp
            ros_img.header.frame_id = 'lidar_link'
            self.lidar_img_pub.publish(ros_img)

    # ── LiDAR → top-down image ────────────────────────────────────────────────

    def render_lidar_bgr(self):
        """Render the current LaserScan as a top-down bird's-eye BGR image.

        Robot at centre, facing UP.  White dots = obstacles.
        Coloured rings mark 3 / 6 / 9 m.
        Returns a numpy BGR uint8 array, or None if no scan available.
        """
        if self.latest_scan is None:
            return None

        scan   = self.latest_scan
        size   = LIDAR_IMG_SIZE
        cx, cy = size // 2, size // 2
        scale  = (size // 2 - 4) / LIDAR_MAX_RANGE  # px per metre

        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Distance rings
        ring_colors = [(120, 60, 60), (60, 120, 60), (60, 60, 120)]  # BGR
        for dist, color in zip(LIDAR_RINGS_M, ring_colors):
            cv2.circle(img, (cx, cy), int(dist * scale), color, 1)

        # Forward direction tick
        cv2.line(img, (cx, cy), (cx, cy - int(LIDAR_MAX_RANGE * scale)),
                 (80, 80, 80), 1)

        # Robot marker (cyan dot)
        cv2.circle(img, (cx, cy), 4, (0, 200, 255), -1)

        # Scan points
        angles = (scan.angle_min
                  + np.arange(len(scan.ranges)) * scan.angle_increment)
        ranges = np.array(scan.ranges, dtype=np.float32)

        valid  = (np.isfinite(ranges)
                  & (ranges > scan.range_min)
                  & (ranges < LIDAR_MAX_RANGE))
        angles = angles[valid]
        ranges = ranges[valid]

        # ROS REP-103: angle 0 = forward (+x), positive CCW = left (+y)
        # Image: forward = up (-row), left = left (-col)
        px_col = (cx - ranges * np.sin(angles) * scale).astype(int)
        px_row = (cy - ranges * np.cos(angles) * scale).astype(int)

        in_bounds = ((px_col >= 0) & (px_col < size)
                     & (px_row >= 0) & (px_row < size))
        for c, r in zip(px_col[in_bounds], px_row[in_bounds]):
            cv2.circle(img, (int(c), int(r)), 2, (255, 255, 255), -1)

        return img

    def render_lidar_base64(self):
        """Return the LiDAR image as a base64 JPEG string for the LLM."""
        bgr = self.render_lidar_bgr()
        if bgr is None:
            return None
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')

    # ── Main LLM loop ─────────────────────────────────────────────────────────

    def process_llm_logic(self):
        if self.latest_image is None:
            self.get_logger().warn("Waiting for camera image...")
            return
        if self.latest_scan is None:
            self.get_logger().warn("Waiting for LiDAR scan...")
            return

        # Encode camera image
        cv_img  = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
        _, buf  = cv2.imencode('.jpg', cv2.resize(cv_img, (320, 240)))
        cam_b64 = base64.b64encode(buf).decode('utf-8')

        lidar_b64 = self.render_lidar_base64()

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.history:
            history_str = " -> ".join([h['action'] for h in self.history])
            messages.append({
                "role": "user",
                "content": f"My last {len(self.history)} actions were: {history_str}."
            })

        if lidar_b64:
            content = [
                {"type": "text",      "text": "IMAGE 1 — Camera (forward view):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
                {"type": "text",      "text": "IMAGE 2 — LiDAR map (top-down, robot at centre facing up):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{lidar_b64}"}},
                {"type": "text",      "text": "What is my next action?"},
            ]
        else:
            content = [
                {"type": "text",      "text": "Camera view only (LiDAR unavailable). What is my next action?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_b64}"}},
            ]

        messages.append({"role": "user", "content": content})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                response_format={"type": "json_object"},
            )

            res       = json.loads(response.choices[0].message.content)
            action    = res['action']
            reasoning = res['reasoning']

            self.history.append({"action": action, "reasoning": reasoning})
            if len(self.history) > self.max_history:
                self.history.pop(0)

            self.publish_commands(action, reasoning, res)

        except Exception as e:
            self.get_logger().error(f"LLM call failed: {e}")
            self.publish_stop()

    # ── Command publishing ────────────────────────────────────────────────────

    def publish_commands(self, action, reasoning, res):
        intent           = AgentIntent()
        intent.intent    = action
        intent.reasoning = reasoning
        self.intent_pub.publish(intent)

        msg = Twist()
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