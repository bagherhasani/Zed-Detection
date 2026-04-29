# Zed-Detection Demo

<video src="https://github.com/user-attachments/assets/6ef8f8bb-d1a6-4df6-95c7-5e376df29d8e" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

# 3D Perception & Autonomous Human-Following Pipeline
### High-Performance Robotics Integration: ROS2 | ZED Stereo | Ouster LiDAR | AgileX Tracer

## 🚀 Overview
This repository showcases an advanced perception-to-control pipeline developed for the [AgileX Tracer](https://global.agilex.ai/products/tracer) mobile base. I engineered a robust **Target Re-Identification (Re-ID)** system that allows the robot to autonomously track and follow a specific human subject while navigating dynamic environments using multi-sensor fusion.

## 🛠 The Technical Stack

### **Hardware & Sensors**
* **Mobile Base:** [AgileX Tracer](https://global.agilex.ai/products/tracer) – A high-speed, dual-wheel differential drive AGV.
* **Vision & Depth:** [Stereolabs ZED](https://www.stereolabs.com/docs/ros2/ros2-robot-integration) – Used for high-resolution stereo depth mapping and 3D spatial coordinate extraction.
* **LiDAR:** [Ouster OS1](https://ouster.com/products/hardware/os1-lidar-sensor) – Integrated for high-fidelity spatial awareness and environmental SLAM.

### **Software & Libraries**
* **Middleware:** **ROS2 (Robot Operating System)** – Orchestrates communication between the camera, LiDAR, and motor controllers.
* **Inference Engine:** **PyTorch** & **torchreid** – Powers the OSNet model for real-time person re-identification.
* **Vision Processing:** **OpenCV (cv2)** – Image manipulation, bounding box logic, and UI overlays.
* **Math & Logic:** `NumPy` & `SciPy` – Used for Euclidean distance calculations and heading geometry.

## 🧠 Engineering Challenges Solved

### 1. Robust Target Re-Identification
Standard color-based trackers fail during occlusions. I implemented an inference-based Re-ID logic that creates a unique feature embedding for the target.
* **The Result:** The robot can distinguish between multiple people and "wait" to re-acquire the specific target once they reappear, preventing accidental tracking of bystanders.

### 2. Multi-Sensor Spatial Fusion
I fused 2D visual detections with 3D point cloud data to generate precise navigation vectors:
* **Distance Calculation:** $d = \sqrt{x^2 + y^2 + z^2}$ using the ZED Depth API.
* **Heading Control:** $\theta = \arctan\left(\frac{x}{z}\right)$ to maintain centering and calculate steering angle.
* **Environment Mapping:** Utilized the Ouster LiDAR via `mapper_params_online_async_ouster.yaml` for simultaneous localization and mapping (SLAM), ensuring the robot avoids obstacles while following.

## 📂 Key Files & Contributions
* **`zed-color.py`**: The core control loop integrating vision, Re-ID, and ROS2 command generation.
* **`nav2_params_tracer_ouster_slam.yaml`**: Custom configuration for the Nav2 stack, optimizing the Ouster LiDAR for the Tracer base.
* **`osnet_x025_reid.onnx`**: The optimized inference model used for high
