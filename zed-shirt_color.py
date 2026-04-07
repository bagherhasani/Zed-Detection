import time

import cv2
import numpy as np
import pyzed.sl as sl


def name_from_hsv(h, s, v):
    """Roughly map HSV to a basic color name."""
    if v < 40:
        return "BLACK / VERY DARK"
    if s < 40:
        if v > 200:
            return "WHITE / VERY LIGHT"
        return "GRAY"

    if h < 10 or h >= 170:
        return "RED"
    if 10 <= h < 25:
        return "ORANGE"
    if 25 <= h < 35:
        return "YELLOW"
    if 35 <= h < 85:
        return "GREEN"
    if 85 <= h < 135:
        return "BLUE"
    if 135 <= h < 170:
        return "PURPLE"
    return "UNKNOWN"


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

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    point_cloud = sl.Mat()

    print("Stand roughly centered in front of the camera.")
    print("Press 'q' to quit.")

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()  # BGRA
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h_img, w_img, _ = frame_bgr.shape

        # Define a torso ROI around the center of the image
        roi_w = int(0.3 * w_img)
        roi_h = int(0.4 * h_img)
        cx = w_img // 2
        cy = int(0.55 * h_img)  # a bit below center

        x1 = max(0, cx - roi_w // 2)
        x2 = min(w_img, cx + roi_w // 2)
        y1 = max(0, cy - roi_h // 2)
        y2 = min(h_img, cy + roi_h // 2)

        roi = frame_bgr[y1:y2, x1:x2]

        color_text = "NO PERSON IN ROI"

        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Only consider reasonably saturated & bright pixels
            h, s, v = cv2.split(hsv)
            mask = (s > 40) & (v > 60)

            if np.any(mask):
                hue_vals = h[mask]

                # Histogram over hue to find dominant color
                hist, bin_edges = np.histogram(hue_vals, bins=36, range=(0, 180))
                if hist.sum() > 0:
                    max_bin = np.argmax(hist)
                    h_center = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2.0

                    # Approximate average s, v in the mask
                    s_mean = int(s[mask].mean())
                    v_mean = int(v[mask].mean())

                    color_name = name_from_hsv(int(h_center), s_mean, v_mean)
                    color_text = f"Shirt color: {color_name}"
            else:
                color_text = "ROI too dark / unsaturated"

        # Draw ROI rectangle
        cv2.rectangle(
            frame_bgr,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
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
            f"ZED 2i Shirt Color ~{fps:.1f} FPS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame_bgr,
            "Stand so your torso is inside the blue box",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame_bgr,
            color_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ZED Shirt Color Demo", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


