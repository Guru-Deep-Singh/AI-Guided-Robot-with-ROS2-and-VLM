## AI-Guided Robot with ROS2 and VLM

[![ROS 2](https://img.shields.io/badge/ROS2-Foxy-blue.svg)](https://docs.ros.org/en/foxy/)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![Gazebo](https://img.shields.io/badge/Gazebo-Classic-orange.svg)](https://gazebosim.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-20.04-E95420.svg)](https://releases.ubuntu.com/20.04/)
[![VLM](https://img.shields.io/badge/Vision%20Language%20Model-LLM%2FVLM-purple.svg)](#)
[![AMD MI300X](https://img.shields.io/badge/AMD%20MI300X-Instinct-ED1C24.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

This project provides a ROS 2-based autonomous driving stack for a simple car robot, controlled by a cloud (or local) Large Language Model / Vision Language Model (LLM/VLM).  
Gazebo is used to simulate the robot in a road environment, and an LLM decides high-level driving actions (FORWARD / LEFT / RIGHT / STOP) from camera images.

## Monocular Vision-Based Guidance with OpenAI GPT4.1
![gif](./images/phase1.gif)

## Monocular and 2D Lidar-Based Guidance with Qwen2.5-VL-72B on AMD MI300x
![gif](./images/phase2.gif)

## Fine-Tuned Qwen2.5-VL-7B for Autonomous Driving
![gif](./images/phase3.gif)

### Repository layout

- **`ros2_ws/`**: ROS 2 workspace
  - **`src/rwi_bringup`**: launch files to start Gazebo, spawn the robot, and run the LLM driver
    - `launch/sim_gazebo.launch.py`: starts Gazebo with `road_uturn.world`, spawns the robot, and launches the agent process (supports `use_lidar` parameter)
    - `launch/stereo_depth.launch.py`: stereo processing pipeline (rectification, disparity, point cloud)
  - **`src/rwi_agent_cloud`**: Python node that talks to the LLM backend
    - `rwi_agent_cloud/llm_driver_node.py`: Camera-only driver node. Subscribes to camera images, calls the LLM (OpenAI or Ollama) and publishes `cmd_vel` and `AgentIntent`.
    - `rwi_agent_cloud/llm_driver_node_lidar.py`: Lidar + Camera driver node. Subscribes to camera and lidar, renders a 2D top-down lidar map, and sends both as multimodal input to the LLM (OpenAI or Local/Qwen).
    - `rwi_agent_cloud/teleop_logger_node.py`: Teleoperation node that allows you to control the robot via arrow keys while collecting supervised driving datasets (camera, lidar projection, keystrokes).
    - `setup.py`: Python packaging / console entry point (`llm_driver`)
  - **`src/rwi_interfaces`**: custom message definitions
    - `msg/AgentIntent.msg`: intent + reasoning from the LLM
  - **`src/rwi_sim_gazebo`**: Gazebo simulation assets (`worlds/road_uturn.world`, etc.)
  - **`src/rwi_description`**: robot description / URDF (`urdf/simple_car.urdf`)

- **`qwen_creation/`**: Setup for hosting local VLMs (specifically Qwen2.5-VL)
  - `qwen_vl_server.py`: FastAPI server exposing an OpenAI-compatible endpoint for Qwen2.5-VL.
  - `qwen_vl_server_finetuned.py`: AMD cloud server script specifically adapted for the fine-tuned model.
  - `test/test_qwen_fine_tuned.py`: Script to test the fine-tuned Qwen model's outputs locally.
  - `Dockerfile`: Container configuration for ROCm/AMD GPU deployment.
  - `Setup_qwen_AMD_Cloud.md`: Deployment instructions for AMD Cloud.

- **`fine_tuning/`**: Scripts and notebooks for fine-tuning the VLM
  - `qwen_robot_finetune_no_quant.ipynb`: Jupyter Notebook containing the Qwen2.5-VL-7B fine-tuning implementation with the personalized teleop dataset.
  - `inspect_dataset.py` & `upload_dataset_to_hf.py`: Utilities to preview the collected teleop log datasets and easily upload them to HuggingFace.

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

- **`LLM_BACKEND`**: `"openai"` or `"local"` (use `"local"` for Ollama or self-hosted servers)
- **OpenAI backend**
  - `OPENAI_MODEL` (default: `gpt-4.1`)
  - `OPENAI_API_KEY`
- **Local/Custom backend (Ollama, Qwen-VL, etc.)**
  - `LOCAL_BASE_URL`: Base URL of your hosted model (e.g. `http://localhost:11434/v1` for Ollama or `http://<cloud-ip>:8001/v1` for a custom server)
  - `LOCAL_API_KEY`: API key if required (defaults to "none")
  - `LOCAL_MODEL_TYPE`: Used to parse prompts correctly depending on the model format (e.g. `general` or `finetuned`)
  - `LOCAL_MODEL`: Model name (e.g. `Qwen/Qwen2.5-VL-72B-Instruct` or `biggestFudge/qwen2-5-vl-7b-robot-merged-v2`)
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
ros2 launch rwi_bringup sim_gazebo.launch.py agent_active:=true use_lidar:=false
```

Parameters:
- `agent_active` (default: `true`): Whether to start the agent node.
- `use_lidar` (default: `false`): 
    - `false`: Launches `llm_driver_node.py` (Camera only).
    - `true`: Launches `llm_driver_node_lidar.py` (Camera + 2D Lidar).

This will:

- Start Gazebo with the `road_uturn.world` map
- Spawn the `simple_car` robot
- Launch the selected LLM driver node (`llm_driver_node` or `llm_driver_node_lidar`), which:
  - Subscribes to camera images (and `/scan` if `use_lidar` is true)
  - Encodes and sends multimodal data to the selected LLM backend
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

### Local VLM Hosting (Qwen2.5-VL)

For high-performance local inference, this repository includes a FastAPI server for **Qwen2.5-VL**, optimized for AMD GPUs (ROCm).

1. **Setup on AMD Cloud / Local Machine:**
   Follow the detailed instructions in [`qwen_creation/Setup_qwen_AMD_Cloud.md`](./qwen_creation/Setup_qwen_AMD_Cloud.md).

2. **Run with Docker:**
   ```bash
   docker build -t qwen-vl:rocm ./qwen_creation
   docker run --rm -it --device=/dev/kfd --device=/dev/dri -p 8001:8000 qwen-vl:rocm
   ```

3. **Configure ROS 2 node:**
   Update your `.env` to use the local backend:
   ```env
   LLM_BACKEND=local
   LOCAL_BASE_URL=http://your-server-ip:8001
   LOCAL_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
   ```

