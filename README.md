# Zed-Detection Demo

<video src="https://github.com/user-attachments/assets/6ef8f8bb-d1a6-4df6-95c7-5e376df29d8e" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>

## 3D Human-Following & Perception Pipeline

This project implements a robust human-following system using a **Tracer mobile robot** and a **ZED Stereo Camera**. Built on **ROS2**, the system is designed to handle real-world challenges like occlusions and frame loss, ensuring the robot consistently tracks and follows a specific target.

### Key Features

* **Person Re-Identification:** The system can re-acquire a specific target even if they leave the frame or are temporarily blocked by other people—something basic color segmentation cannot achieve.
* **3D Spatial Perception:** Maps 2D detections into 3D space using the ZED SDK to provide accurate $(x, y, z)$ coordinates.
* **Autonomous Navigation:** Automatically calculates the required velocity and heading to keep the target centered at a safe distance.
* **Hardware Integration:** Directly interfaces with the Tracer mobile base via ROS2 `Twist` messages.

### Technical Implementation

The pipeline processes live frames to identify a specific human target and fuses that data with ZED stereo depth for 3D localization.

* **Distance Calculation:** $d = \sqrt{x^2 + y^2 + z^2}$
* **Heading Angle:** $\theta = \arctan\left(\frac{x}{z}\right)$

These values are fed into a control loop that commands `linear.x` and `angular.z` to the robot base.

### Tech Stack

* **Framework:** ROS2
* **Vision & Depth:** ZED SDK, OpenCV
* **Inference:** PyTorch
* **Hardware:** Tracer Mobile Robot, ZED Camera

### Usage

To run the primary detection and following script:

```bash
python3 zed-color.py
