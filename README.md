## AI-Guided Robot with ROS2 and VLM

[![ROS 2](https://img.shields.io/badge/ROS2-Foxy-blue.svg)](https://docs.ros.org/en/foxy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic-orange.svg)](https://gazebosim.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)](https://releases.ubuntu.com/20.04/)
[![Ollama](https://img.shields.io/badge/Ollama-supported-000000.svg)](https://ollama.com/)
[![VLM](https://img.shields.io/badge/Vision%20Language%20Model-LLM%2FVLM-purple.svg)](#)

![gif](./images/phase1.gif)

This project provides a ROS 2-based autonomous driving stack for a simple car robot, controlled by a cloud (or local) Large Language Model / Vision Language Model (LLM/VLM).  
Gazebo is used to simulate the robot in a road environment, and an LLM decides high-level driving actions (FORWARD / LEFT / RIGHT / STOP) from camera images.

### Repository layout

- **`ros2_ws/`**: ROS 2 workspace
  - **`src/rwi_bringup`**: launch files to start Gazebo, spawn the robot, and run the LLM driver
    - `launch/sim_gazebo.launch.py`: starts Gazebo with `road_uturn.world`, spawns the robot, and launches the agent process
    - `launch/stereo_depth.launch.py`: stereo processing pipeline (rectification, disparity, point cloud)
  - **`src/rwi_agent_cloud`**: Python node that talks to the LLM backend
    - `rwi_agent_cloud/llm_driver_node.py`: subscribes to camera images, calls the LLM (OpenAI or Ollama) and publishes `cmd_vel` and `AgentIntent`
    - `setup.py`: Python packaging / console entry point (`llm_driver`)
  - **`src/rwi_interfaces`**: custom message definitions
    - `msg/AgentIntent.msg`: intent + reasoning from the LLM
  - **`src/rwi_sim_gazebo`**: Gazebo simulation assets (`worlds/road_uturn.world`, etc.)
  - **`src/rwi_description`**: robot description / URDF (`urdf/simple_car.urdf`)

- **`.env.example`**: example environment variables for LLM configuration (copy to `.env` and edit)

### Prerequisites

- **ROS 2 Foxy** (tested) or a compatible distro with common desktop packages:
  - `gazebo_ros`, `robot_state_publisher`, `image_proc`, `stereo_image_proc`, `rclcpp_components`, etc.
- **Python 3.8+** with:
  - `python-dotenv`
  - `openai`
  - `opencv-python` (plus ROS image bridge `cv_bridge`, via ROS packages)

You may want a dedicated virtual environment for the Python dependencies that the LLM node uses.

### Environment configuration

Copy the example environment file at the repo root and adjust it:

```bash
cp .env.example .env
```

Relevant variables:

- **`LLM_BACKEND`**: `"openai"` or `"ollama"`
- **OpenAI backend**
  - `OPENAI_MODEL` (default: `gpt-4.1`)
  - `OPENAI_API_KEY`
- **Ollama backend**
  - `OLLAMA_BASE_URL` (e.g. `http://localhost:11434/v1` or a cloud endpoint)
  - `OLLAMA_API_KEY` (if your Ollama endpoint requires it)
  - `OLLAMA_MODEL`
- **Timing and image options**
  - `LLM_INTERVAL` (seconds between LLM calls, default `1.5`)
  - `IMAGE_SIZE` (e.g. `320x240`)
  - `JPEG_QUALITY` (0–100, trade-off between size and quality)

The `llm_driver_node.py` will:

1. Call `load_dotenv()` to load variables from the current working directory / parent directories.
2. If `OPENAI_API_KEY` and `LLM_BACKEND` are still unset, it will explicitly load `.env` from the **project root** (this directory).

### Building the ROS 2 workspace

From the repository root:

```bash
cd ros2_ws
colcon build
```

Then source the workspace (adapt the ROS 2 distro and shell as needed):

```bash
source install/setup.bash
```

### Running the simulation with the LLM driver

After building and sourcing the workspace:

```bash
cd ros2_ws
ros2 launch rwi_bringup sim_gazebo.launch.py agent_active:=true
```

This will:

- Start Gazebo with the `road_uturn.world` map
- Spawn the `simple_car` robot
- Launch the LLM driver node (`rwi_agent_cloud.llm_driver_node`), which:
  - Subscribes to `/car_camera/image_raw`
  - Encodes and sends the image to the selected LLM backend
  - Receives an action decision and reasoning as JSON
  - Publishes:
    - `AgentIntent` on `/agent/intent`
    - `Twist` commands on `/cmd_vel` to drive the robot

You should see log messages in the terminal indicating the chosen action, perceived scene, and short reasoning.

### Stereo depth processing (optional)

To enable stereo processing (rectification, disparity, and point cloud) in the simulation:

```bash
cd ros2_ws
ros2 launch rwi_bringup stereo_depth.launch.py
```

This launches composable nodes for:

- Left/right image rectification
- Disparity computation
- Point cloud generation (`/stereo_camera/points2`)

### Development notes

- The `rwi_agent_cloud` package exposes `llm_driver` as a console script entry point, mapping to `rwi_agent_cloud.llm_driver_node:main`.
- The node supports two backends via the `LLM_BACKEND` variable:
  - `"ollama"`: uses `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` / `OLLAMA_MODEL`
  - `"openai"` (default): uses `OPENAI_API_KEY` / `OPENAI_MODEL`
- The logic in `llm_driver_node.py` includes a detailed system prompt and maintains a short history of past actions to reduce oscillations in steering decisions.

