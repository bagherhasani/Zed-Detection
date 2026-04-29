# Zed-Detection Demo

<video src="https://github.com/user-attachments/assets/6ef8f8bb-d1a6-4df6-95c7-5e376df29d8e" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>


# 3D Perception & Autonomous Human-Following Pipeline
### High-Performance Robotics Integration: ROS2 | ZED Stereo | Ouster LiDAR | Tracer Base

## 🚀 Overview
This repository showcases an advanced perception-to-control pipeline developed for the **Tracer mobile robot**. Unlike standard tracking scripts, this project implements a production-grade approach to **Target Re-Identification (Re-ID)** and **Multi-Sensor Fusion**, allowing a robot to autonomously track and follow a specific human subject in dynamic, cluttered environments.

## 🛠 Technical Showpiece: The Stack
I integrated a diverse hardware and software stack to solve the "Lost-Target" problem in mobile robotics:

* **Vision & Depth:** [Stereolabs ZED SDK](https://www.stereolabs.com/docs/) & Camera for stereo depth mapping and spatial coordinates.
* **LiDAR Integration:** [Ouster OS1](https://ouster.com/products/hardware/os1-lidar-sensor) for high-fidelity spatial awareness and asynchronous SLAM.
* **Middleware:** **ROS2 (Robot Operating System)** for modular communication and hardware abstraction.
* **Deep Learning:** **PyTorch** powered inference for human detection and identity embedding.
* **Hardware:** **Tracer Mobile Base** controlled via real-time `Twist` command interfaces.

## 🧠 Complex Challenges Solved

### 1. Robust Target Re-Identification
Standard color-based trackers fail when a target is occluded or leaves the frame. I implemented an inference-based Re-ID logic that creates a unique feature embedding for the target.
* **The Result:** The robot can distinguish between multiple people and "wait" to re-acquire the specific target once they reappear, preventing accidental tracking of bystanders.

### 2. Multi-Sensor Spatial Fusion
I combined 2D visual detection with 3D point cloud data to compute precise navigation vectors.
* **Distance Logic:** $d = \sqrt{x^2 + y^2 + z^2}$ using ZED Depth API.
* **Heading Control:** $\theta = \arctan\left(\frac{x}{z}\right)$ to maintain centering.
* **Environment Mapping:** Utilizing the **Ouster LiDAR** via `mapper_params_online_async_ouster.yaml` for simultaneous localization and mapping (SLAM), ensuring the robot understands its surroundings while following.

## 📂 Repository Structure & Contributions
* **`zed-color.py`**: The core control loop integrating vision, Re-ID, and ROS2 command generation.
* **`nav2_params_tracer_ouster_slam.yaml`**: Custom configuration for the Nav2 stack, optimizing the Ouster LiDAR for the Tracer base.
* **`prepare_reid_model.py`**: Script for optimizing the deep learning model for real-time edge inference.
* **`osnet_x025_reid.onnx`**: The optimized inference model used for person identification.

## 📈 Impact & Performance
* **Hardware-Tested:** Successfully deployed on physical hardware (Tracer + ZED + Ouster).
* **Real-Time Latency:** Optimized the perception pipeline to maintain a steady control frequency, essential for smooth robot movement.
* **Full-Stack Ownership:** Designed everything from the low-level ROS2 `Twist` messaging to high-level deep learning inference.

## 🛠 Installation
```bash
# Clone and source ROS2 environment
git clone [https://github.com/bagherhasani/Zed-Detection.git](https://github.com/bagherhasani/Zed-Detection.git)
# Run the integrated following pipeline
python3 zed-color.py
