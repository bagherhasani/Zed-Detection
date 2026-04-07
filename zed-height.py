import math
import time

import cv2
import numpy as np
import pyzed.sl as sl


def main():
    # --- Init ZED camera ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", status)
        return

    # Enable positional tracking (required for body tracking)
    tracking_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking")
        zed.close()
        return

    # Body tracking setup
    body_params = sl.BodyTrackingParameters()
    body_params.enable_tracking = True
    body_params.enable_body_fitting = True
    body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_params.body_format = sl.BODY_FORMAT.BODY_38

    if zed.enable_body_tracking(body_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable body tracking")
        zed.close()
        return

    body_runtime = sl.BodyTrackingRuntimeParameters()

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    print("Stand in front of the camera. Press 'q' to quit.")
    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()  # BGRA
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Retrieve detected bodies
        bodies = sl.Bodies()
        zed.retrieve_bodies(bodies, body_runtime)

        msg = "No person detected"

        if bodies.is_new and len(bodies.body_list) > 0:
            # Take the first tracked body
            person = None
            for b in bodies.body_list:
                if b.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    person = b
                    break

            if person is not None:
                # Draw 2D bounding box
                bb2d = np.array(person.bounding_box_2d, dtype=np.int32)
                cv2.polylines(frame_bgr, [bb2d.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

                # Estimate height from 3D keypoints (head to feet)
                keypoints = np.array(person.keypoint)
                # BODY_PARTS indices: use HEAD and average of feet/ankles
                try:
                    head_idx = sl.BODY_PARTS.HEAD.value
                    l_ankle_idx = sl.BODY_PARTS.LEFT_ANKLE.value
                    r_ankle_idx = sl.BODY_PARTS.RIGHT_ANKLE.value

                    head = keypoints[head_idx]
                    feet = (keypoints[l_ankle_idx] + keypoints[r_ankle_idx]) / 2.0

                    if not np.any(np.isnan(head)) and not np.any(np.isnan(feet)):
                        height_m = float(np.linalg.norm(head - feet))
                        height_cm = height_m * 100.0

                        # Distance from camera (use feet position)
                        dist_m = float(math.sqrt(feet[0] ** 2 + feet[2] ** 2))

                        # Check distance range for reliable measurement
                        if dist_m < 1.2:
                            msg = "Too close, step back a bit"
                        elif dist_m > 3.0:
                            msg = "Too far, step closer"
                        else:
                            msg = f"Estimated height: {height_cm:.1f} cm"

                        # Overlay info near top of bounding box
                        x, y = int(bb2d[0][0]), int(bb2d[0][1])
                        cv2.putText(
                            frame_bgr,
                            msg,
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        msg = "Body keypoints not reliable"
                except Exception:
                    msg = "Body tracking model not available"

        # Show generic message if nothing else wrote one
        if msg and "Estimated height" not in msg and "Too" not in msg:
            cv2.putText(
                frame_bgr,
                msg,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # FPS
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0

        cv2.putText(
            frame_bgr,
            f"ZED 2i Body Tracking ~{fps:.1f} FPS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("ZED Person Height Demo", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


