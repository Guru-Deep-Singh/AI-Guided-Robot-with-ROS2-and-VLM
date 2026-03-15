#!/usr/bin/env python3
"""
teleop_logger_node.py
─────────────────────
Keyboard teleoperation + dataset logger for fine-tuning data collection.
Logs synchronised (camera image, LiDAR top-down render, intent) triples.

LiDAR rendering is taken directly from llm_driver_node_lidar.py so the
logged lidar_b64 is pixel-identical to what the fine-tuned model will
receive at inference time — guaranteeing train/inference consistency.

No API calls are made. This node is purely for data collection.

Controls (terminal must have focus):
  ↑  Arrow Up    → FORWARD
  ↓  Arrow Down  → REVERSE
  ←  Arrow Left  → LEFT
  →  Arrow Right → RIGHT
  Space          → STOP
  L              → toggle logging ON / OFF
  Q              → quit

Each saved sample (JSONL, one object per line):
  {
    "image_b64":  "<camera JPEG, base64>",
    "lidar_b64":  "<top-down LiDAR JPEG, base64>",
    "intent":     "FORWARD | LEFT | RIGHT | REVERSE | STOP"
  }

ROS topics consumed:
  /car_camera/image_raw   (sensor_msgs/Image)     ← same as llm_driver_node_lidar
  /scan                   (sensor_msgs/LaserScan)  ← same as llm_driver_node_lidar

ROS topics published:
  /cmd_vel                (geometry_msgs/Twist)
  /llm/lidar_image        (sensor_msgs/Image)      ← live preview, same as driver node

Parameters (--ros-args -p key:=value):
  image_topic    str    default "/car_camera/image_raw"
  scan_topic     str    default "/scan"
  cmd_vel_topic  str    default "/cmd_vel"
  output_dir     str    default "~/dataset"
  log_interval   float  default 0.5   seconds between saved samples
  linear_speed   float  default 0.3   m/s   (matches local-backend driver)
  angular_speed  float  default 0.05  rad/s (matches local-backend driver)
  image_width    int    default 320
  image_height   int    default 320   kept square like driver node default
  jpeg_quality   int    default 85

Usage:
  ros2 run rwi_agent_cloud teleop_logger
"""

import os
import sys
import json
import time
import base64
import threading
import termios
import tty
import select
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

# ──────────────────────────────────────────────────────────────────────────────
# LiDAR render constants — kept identical to llm_driver_node_lidar.py
# so logged lidar images are pixel-perfect matches to inference-time images.
# ──────────────────────────────────────────────────────────────────────────────
LIDAR_IMG_SIZE  = 256
LIDAR_MAX_RANGE = 12.0
LIDAR_FOV_DEG   = 120.0          # −60° to +60°, matches URDF
LIDAR_RINGS_M   = [4.0, 8.0, 12.0]

# ──────────────────────────────────────────────────────────────────────────────
# Key-code constants (ANSI escape sequences + regular chars)
# ──────────────────────────────────────────────────────────────────────────────
KEY_UP    = '\x1b[A'
KEY_DOWN  = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT  = '\x1b[D'
KEY_SPACE = ' '
KEY_L     = 'l'
KEY_L_CAP = 'L'
KEY_Q     = 'q'
KEY_Q_CAP = 'Q'

INTENT_MAP = {
    KEY_UP:    'FORWARD',
    KEY_DOWN:  'REVERSE',
    KEY_LEFT:  'LEFT',
    KEY_RIGHT: 'RIGHT',
    KEY_SPACE: 'STOP',
}


# ──────────────────────────────────────────────────────────────────────────────
# Non-blocking keyboard reader
# ──────────────────────────────────────────────────────────────────────────────

def read_key(timeout: float = 0.05) -> str:
    """
    Read one keypress (including 3-byte ANSI arrow sequences) from stdin.
    Returns '' if nothing pressed within timeout.
    """
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ''
    ch = os.read(fd, 1).decode('utf-8', errors='ignore')
    if ch == '\x1b':                             # start of escape sequence
        rlist2, _, _ = select.select([sys.stdin], [], [], 0.02)
        if rlist2:
            ch += os.read(fd, 2).decode('utf-8', errors='ignore')
    return ch


# ──────────────────────────────────────────────────────────────────────────────
# Main node
# ──────────────────────────────────────────────────────────────────────────────

class TeleopLoggerNode(Node):

    def __init__(self):
        super().__init__('teleop_logger_node')

        # ── declare & read parameters ─────────────────────────────────────────
        self.declare_parameter('image_topic',   '/car_camera/image_raw')
        self.declare_parameter('scan_topic',    '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('output_dir',    str(Path.home() / 'dataset'))
        self.declare_parameter('log_interval',  0.5)
        self.declare_parameter('linear_speed',  0.5) # 0.3
        self.declare_parameter('linear_speed_while_rotation',  0.3)
        self.declare_parameter('angular_speed', 0.05)
        self.declare_parameter('image_width',   320)
        self.declare_parameter('image_height',  320)
        self.declare_parameter('jpeg_quality',  85)

        self.image_topic   = self.get_parameter('image_topic').value
        self.scan_topic    = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.output_dir    = Path(self.get_parameter('output_dir').value).expanduser()
        self.log_interval  = self.get_parameter('log_interval').value
        self.linear_speed  = self.get_parameter('linear_speed').value
        self.linear_speed_while_rotation = self.get_parameter('linear_speed_while_rotation').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.img_w         = self.get_parameter('image_width').value
        self.img_h         = self.get_parameter('image_height').value
        self.jpeg_quality  = self.get_parameter('jpeg_quality').value

        # ── shared state ──────────────────────────────────────────────────────
        self.bridge         = CvBridge()
        self.current_intent = 'STOP'
        self.current_image  = None          # latest cv2 BGR camera frame
        self.latest_scan    = None          # latest LaserScan msg
        self.data_lock      = threading.Lock()
        self.logging_active = False
        self.sample_count   = 0
        self.last_log_time  = 0.0

        # ── dataset output file ───────────────────────────────────────────────
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.jsonl_path = self.output_dir / f'dataset_{ts}.jsonl'
        self.jsonl_file = open(self.jsonl_path, 'a', encoding='utf-8')
        self.get_logger().info(f'Dataset file: {self.jsonl_path}')

        # ── ROS publishers ────────────────────────────────────────────────────
        self.cmd_pub       = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.lidar_img_pub = self.create_publisher(Image, '/llm/lidar_image',  10)

        # ── ROS subscribers ───────────────────────────────────────────────────
        self.create_subscription(Image,     self.image_topic, self._image_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic,  self._scan_cb,  10)

        # 50 Hz control loop — keeps /cmd_vel alive while key held
        self.create_timer(0.02, self._control_loop)
        # Logging check at 10 Hz, actual save gated by log_interval
        self.create_timer(0.1,  self._logging_loop)

        # ── keyboard thread ───────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._kbd_thread = threading.Thread(target=self._keyboard_loop, daemon=True)

        self._print_banner()

    # ──────────────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        """Store latest camera frame, resized to configured resolution."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.resize(frame, (self.img_w, self.img_h))
            with self.data_lock:
                self.current_image = frame
        except Exception as e:
            self.get_logger().warn(f'Image conversion error: {e}')

    def _scan_cb(self, msg: LaserScan):
        """Store latest scan and publish live LiDAR preview."""
        with self.data_lock:
            self.latest_scan = msg
        # Publish live preview — mirrors llm_driver_node_lidar behaviour
        try:
            bgr = self._render_lidar_bgr(msg)
            if bgr is not None:
                ros_img = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                ros_img.header.stamp    = msg.header.stamp
                ros_img.header.frame_id = 'lidar_link'
                self.lidar_img_pub.publish(ros_img)
        except Exception as e:
            self.get_logger().warn(f'LiDAR preview publish failed: {e}',
                                   throttle_duration_sec=5.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Timer callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _control_loop(self):
        """Publish Twist at 50 Hz so the robot responds immediately to keys."""
        self.cmd_pub.publish(self._intent_to_twist(self.current_intent))

    def _logging_loop(self):
        """Save one sample when logging is active and interval has elapsed."""
        if not self.logging_active:
            return
        now = time.monotonic()
        if now - self.last_log_time < self.log_interval:
            return

        with self.data_lock:
            frame = self.current_image
            scan  = self.latest_scan

        if frame is None:
            self.get_logger().warn('No camera image yet — skipping sample.',
                                   throttle_duration_sec=3.0)
            return
        if scan is None:
            self.get_logger().warn('No LiDAR scan yet — skipping sample.',
                                   throttle_duration_sec=3.0)
            return

        lidar_bgr = self._render_lidar_bgr(scan)
        if lidar_bgr is None:
            self.get_logger().warn('LiDAR render returned None — skipping sample.',
                                   throttle_duration_sec=3.0)
            return

        self._save_sample(frame, lidar_bgr, self.current_intent)
        self.last_log_time = now

    # ──────────────────────────────────────────────────────────────────────────
    # Keyboard loop  (own thread — never blocks ROS spin)
    # ──────────────────────────────────────────────────────────────────────────

    def _keyboard_loop(self):
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_event.is_set():
                key = read_key(timeout=0.05)
                if not key:
                    continue

                if key in (KEY_Q, KEY_Q_CAP):
                    self.get_logger().info('Quit requested.')
                    self._stop_event.set()
                    rclpy.shutdown()
                    break

                elif key in (KEY_L, KEY_L_CAP):
                    self.logging_active = not self.logging_active
                    state = 'ON  🔴 recording' if self.logging_active else 'OFF ⬛ paused'
                    self.get_logger().info(
                        f'Logging: {state}  |  samples so far: {self.sample_count}')

                elif key in INTENT_MAP:
                    self.current_intent = INTENT_MAP[key]

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ──────────────────────────────────────────────────────────────────────────
    # LiDAR renderer
    # Exact copy of render_lidar_bgr() from llm_driver_node_lidar.py.
    # Kept as a @staticmethod so it can be unit-tested without a ROS node.
    # DO NOT modify without mirroring the change in llm_driver_node_lidar.py.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render_lidar_bgr(scan: LaserScan):
        """
        Render a 2D LaserScan (120 deg FOV) as a top-down BGR image.
        Robot = cyan dot at centre facing UP.  White dots = obstacles.
        Coloured rings = 4 / 8 / 12 m.  Gray wedge = sensor FOV cone.
        """
        if scan is None:
            return None

        size   = LIDAR_IMG_SIZE
        cx, cy = size // 2, size // 2
        scale  = (size // 2 - 4) / LIDAR_MAX_RANGE

        img = np.zeros((size, size, 3), dtype=np.uint8)

        # 120° coverage wedge
        half_fov = np.radians(LIDAR_FOV_DEG / 2.0)
        r_px     = int(LIDAR_MAX_RANGE * scale)
        n_pts    = 32
        wedge_a  = np.linspace(-half_fov, half_fov, n_pts)
        wx = (cx + r_px * np.sin(wedge_a)).astype(int)
        wy = (cy - r_px * np.cos(wedge_a)).astype(int)
        wedge_pts = np.vstack([[cx, cy],
                                np.column_stack([wx, wy]),
                                [cx, cy]])
        cv2.fillPoly(img, [wedge_pts.astype(np.int32)], (30, 30, 30))

        # Distance rings
        ring_colors = [(100, 60, 60), (60, 100, 60), (60, 60, 100)]
        for dist, color in zip(LIDAR_RINGS_M, ring_colors):
            cv2.circle(img, (cx, cy), int(dist * scale), color, 1)

        # Forward tick + FOV boundary lines
        cv2.line(img, (cx, cy), (cx, cy - r_px), (80, 80, 80), 1)
        for sign in [-1, 1]:
            ex = int(cx + r_px * np.sin(sign * half_fov))
            ey = int(cy - r_px * np.cos(sign * half_fov))
            cv2.line(img, (cx, cy), (ex, ey), (60, 60, 60), 1)

        # Zone labels
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = 0.35
        fc    = (180, 180, 180)
        thick = 1
        cv2.putText(img, "AHEAD", (cx - 22, 12),          font, fs, fc, thick)
        cv2.putText(img, "LEFT",  (4, cy),                 font, fs, fc, thick)
        cv2.putText(img, "RIGHT", (size - 38, cy),         font, fs, fc, thick)
        cv2.putText(img, "4m",  (cx + 6, cy - int(4.0  * scale)), font, fs, (80,80,80), thick)
        cv2.putText(img, "8m",  (cx + 6, cy - int(8.0  * scale)), font, fs, (80,80,80), thick)
        cv2.putText(img, "12m", (cx + 6, cy - int(11.5 * scale)), font, fs, (80,80,80), thick)

        # Robot marker (cyan dot)
        cv2.circle(img, (cx, cy), 4, (0, 200, 255), -1)

        # Scan points — ROS REP-103: angle 0 = forward, positive CCW = left
        angles = (scan.angle_min
                  + np.arange(len(scan.ranges)) * scan.angle_increment)
        ranges = np.array(scan.ranges, dtype=np.float32)
        valid  = (np.isfinite(ranges)
                  & (ranges > scan.range_min)
                  & (ranges < LIDAR_MAX_RANGE))
        angles = angles[valid]
        ranges = ranges[valid]

        px_col = (cx - ranges * np.sin(angles) * scale).astype(int)
        px_row = (cy - ranges * np.cos(angles) * scale).astype(int)
        in_bounds = ((px_col >= 0) & (px_col < size)
                     & (px_row >= 0) & (px_row < size))
        for c, r in zip(px_col[in_bounds], px_row[in_bounds]):
            cv2.circle(img, (int(c), int(r)), 2, (255, 255, 255), -1)

        return img

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _intent_to_twist(self, intent: str) -> Twist:
        """
        Velocity values mirror the local-backend block in llm_driver_node_lidar
        so the robot behaves identically during collection and inference.
        """
        t = Twist()
        if intent == 'FORWARD':
            t.linear.x  =  self.linear_speed
            t.angular.z =  0.0
        elif intent == 'REVERSE':
            t.linear.x  = -self.linear_speed
            t.angular.z =  0.0
        elif intent == 'LEFT':
            t.linear.x  =  self.linear_speed_while_rotation  * 0.5   # ~0.15 m/s forward while turning
            t.angular.z =  self.angular_speed
        elif intent == 'RIGHT':
            t.linear.x  =  self.linear_speed_while_rotation  * 0.5
            t.angular.z = -self.angular_speed
        # STOP → all zeros (Twist default)
        return t

    def _encode_jpeg(self, frame: np.ndarray) -> str:
        """Encode a BGR ndarray as a base64 JPEG string."""
        ok, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def _save_sample(self, camera_frame: np.ndarray,
                     lidar_bgr: np.ndarray, intent: str):
        """Encode both images and append one JSONL record to the dataset file."""
        try:
            cam_b64   = self._encode_jpeg(camera_frame)
            lidar_b64 = self._encode_jpeg(lidar_bgr)
        except RuntimeError as e:
            self.get_logger().warn(f'Encoding failed, skipping sample: {e}')
            return

        record = {
            'image_b64': cam_b64,
            'lidar_b64': lidar_b64,
            'intent':    intent,
        }
        self.jsonl_file.write(json.dumps(record) + '\n')
        self.jsonl_file.flush()
        self.sample_count += 1

        if self.sample_count % 50 == 0:
            self.get_logger().info(
                f'[Logger] {self.sample_count} samples saved → {self.jsonl_path.name}')

    def _print_banner(self):
        self.get_logger().info(
            '\n'
            '╔══════════════════════════════════════════════╗\n'
            '║   Teleop + Dataset Logger                    ║\n'
            '╠══════════════════════════════════════════════╣\n'
            '║  ↑ / ↓ / ← / →   Drive the robot            ║\n'
            '║  SPACE            STOP                       ║\n'
            '║  L                Toggle logging ON / OFF    ║\n'
            '║  Q                Quit                       ║\n'
            '╠══════════════════════════════════════════════╣\n'
            f'║  Camera : {self.image_topic:<35}║\n'
            f'║  Scan   : {self.scan_topic:<35}║\n'
            f'║  LiDAR  : /llm/lidar_image                        ║\n'
            f'║  Output : {str(self.jsonl_path):<35}║\n'
            f'║  Rate   : 1 sample every {self.log_interval}s'
            f'{"":>{14 - len(str(self.log_interval))}}║\n'
            '╚══════════════════════════════════════════════╝'
        )

    def start_keyboard(self):
        self._kbd_thread.start()

    def destroy_node(self):
        """Graceful shutdown: stop robot, flush dataset file."""
        self._stop_event.set()
        # Guard: if Q was pressed, rclpy.shutdown() already ran and the
        # publisher context is invalid. Use print() for the same reason.
        if rclpy.ok():
            self.cmd_pub.publish(Twist())
        self.jsonl_file.flush()
        self.jsonl_file.close()
        print(f'[teleop_logger] Dataset closed. '
              f'Total samples: {self.sample_count} -> {self.jsonl_path}')
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TeleopLoggerNode()
    node.start_keyboard()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()