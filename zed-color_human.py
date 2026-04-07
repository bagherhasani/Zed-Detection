import time
import math

import cv2
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ZedFollower(Node):
    def __init__(self):
        super().__init__("zed_human_follower")

        # Publisher to Tracer base velocity command
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Control parameters (very conservative defaults)
        # --- You can safely tweak these numbers ---
        self.target_distance = 1.2   # desired distance to target [m]
        self.min_distance = 0.7      # stop if closer than this [m]
        self.max_lin_speed = 0.2    # maximum forward speed [m/s]
        self.max_ang_speed = 0.2    # maximum turn speed [rad/s]
        self.k_lin = 0.2             # linear gain  (smaller = slower accel)
        self.k_ang = 0.2             # angular gain (smaller = slower turn)

        # For on-screen feedback
        self.last_lin = 0.0          # last commanded linear speed
        self.last_ang = 0.0          # last commanded angular speed
        self.last_status = "Idle"    # human-readable robot state

        # ZED related
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # or ULTRA if slow
        init_params.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise RuntimeError("Cannot open ZED camera")

        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()

        # Kernel for smoothing the mask
        self.kernel = np.ones((5, 5), np.uint8)

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.get_logger().info(
            "ZED human follower started. Press 'q' or ESC in the window to quit."
        )

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.last_lin = 0.0
        self.last_ang = 0.0
        self.last_status = "STOPPED (no target)"

    def control_robot(self, distance, angle_rad):
        """
        Simple follow controller:
        - Keep target_distance from object
        - Turn to keep object centered
        """
        twist = Twist()

        # Angular control (yaw)
        twist.angular.z = self.k_ang * angle_rad
        twist.angular.z = max(min(twist.angular.z, self.max_ang_speed), -self.max_ang_speed)

        # Linear control (forward)
        if distance is None or distance < self.min_distance:
            twist.linear.x = 0.0
            status = "TOO CLOSE - HOLDING"
        else:
            error_d = distance - self.target_distance
            twist.linear.x = self.k_lin * error_d
            twist.linear.x = max(min(twist.linear.x, self.max_lin_speed), -self.max_lin_speed)
            if abs(twist.linear.x) < 1e-3 and abs(twist.angular.z) < 1e-3:
                status = "ALIGNED - HOLDING"
            else:
                status = "FOLLOWING TARGET"

        self.last_lin = float(twist.linear.x)
        self.last_ang = float(twist.angular.z)
        self.last_status = status
        self.cmd_pub.publish(twist)


def main():
    # --- Init ZED camera ---
    rclpy.init()
    try:
        node = ZedFollower()
    except RuntimeError:
        rclpy.shutdown()
        return

    zed = node.zed
    runtime_params = node.runtime_params
    image = node.image
    point_cloud = node.point_cloud
    kernel = node.kernel

    prev_time = node.prev_time
    frame_count = node.frame_count
    fps = node.fps

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()  # BGRA from ZED SDK
            # Convert BGRA -> BGR for OpenCV (otherwise red/blue are swapped)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # --- Shirt color detection (in HSV space) ---
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

            # Detect a light shirt (off‑white / light gray / pale color).
            # You said your shirt is not pure white, so allow:
            #  - low-to-medium saturation:   S in [0, 90]
            #  - medium-to-high brightness:  V in [130, 255]
            # H (hue) can be anything.
            lower_shirt = np.array([0, 0, 130])
            upper_shirt = np.array([179, 90, 255])

            mask = cv2.inRange(hsv, lower_shirt, upper_shirt)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            detected = False
            if contours:
                # Take the largest contour as the main shirt region
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > 1500:  # ignore tiny noise and very small blobs
                    x, y, w, h = cv2.boundingRect(largest)
                    cx = x + w // 2
                    cy = y + h // 2

                    img_h, img_w, _ = frame_bgr.shape

                    # Heuristics to prefer a human torso:
                    #  - Center of bbox is between 25% and 85% of image height
                    #  - Bbox is tall-ish: height > width * 0.7
                    #  - Bbox is not huge: height < 90% of image height
                    aspect_ok = h > 0.7 * w
                    center_ok = 0.25 * img_h < cy < 0.85 * img_h
                    size_ok = 0.15 * img_h < h < 0.9 * img_h

                    if center_ok and aspect_ok and size_ok:
                    # Retrieve 3D point at the center of the detected region
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                    err, point = point_cloud.get_value(cx, cy)

                    dist_text = ""
                    if err == sl.ERROR_CODE.SUCCESS and point is not None:
                        X, Y, Z, _ = point  # meters
                        distance = (X**2 + Y**2 + Z**2) ** 0.5

                        # Horizontal angle relative to camera forward (Z axis).
                            # Define an error so that positive means "target on LEFT".
                        angle_raw = math.atan2(X, Z)
                        angle_error = -angle_raw
                        angle_err_deg = math.degrees(angle_error)

                        if angle_err_deg > 3:
                            dir_text = "left"
                        elif angle_err_deg < -3:
                            dir_text = "right"
                        else:
                            dir_text = "center"

                        dist_text = (
                            f"{distance:.2f} m, {abs(angle_err_deg):.1f} deg {dir_text}"
                        )

                        # Send follow command to Tracer using the angle error
                        node.control_robot(distance, angle_error)

                    cv2.rectangle(
                        frame_bgr,
                        (x, y),
                        (x + w, y + h),
                            (255, 0, 0),  # rectangle around torso
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                            "TARGET SHIRT DETECTED",
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    if dist_text:
                        cv2.putText(
                            frame_bgr,
                            dist_text,
                            (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    detected = True

            if not detected:
                # No target → stop robot
                node.stop_robot()

                cv2.putText(
                    frame_bgr,
                    "No target shirt",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # FPS estimate (overlaid in the top-left)
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            # Robot status & speeds
            status_text = f"State: {node.last_status}"
            speed_text = f"v={node.last_lin:.2f} m/s, w={node.last_ang:.2f} rad/s"

            cv2.putText(
                frame_bgr,
                status_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame_bgr,
                speed_text,
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Info text
            cv2.putText(
                frame_bgr,
                f"ZED 2i - ~{fps:.1f} FPS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame_bgr,
                "Press Q or ESC to STOP",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow("ZED Blue Detection", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break

    # On exit: stop robot and clean up
    node.stop_robot()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()