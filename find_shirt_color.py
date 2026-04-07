"""
Shirt color finder using ZED camera.

- Opens the ZED camera live
- Press SPACE to freeze/unfreeze the frame
- Click on your shirt to get HSV values + ready-to-paste range
- Press 's' to save the frozen frame as shirt_snapshot.jpg
- Press 'q' or ESC to quit
"""

import sys
import cv2
import numpy as np
import pyzed.sl as sl


# ── state shared with mouse callback ──────────────────────────────────────────
state = {
    "frame": None,
    "hsv":   None,
    "frozen": False,
}


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if state["frame"] is None:
        return

    frame = state["frame"]
    hsv   = state["hsv"]

    bgr = frame[y, x].tolist()
    h, s, v = int(hsv[y, x, 0]), int(hsv[y, x, 1]), int(hsv[y, x, 2])

    h_lo = max(0,   h - 20)
    h_hi = min(179, h + 20)
    s_lo = max(0,   s - 60)
    v_lo = max(0,   v - 60)

    print(f"\n--- Clicked pixel ({x}, {y}) ---")
    print(f"  BGR : B={bgr[0]}  G={bgr[1]}  R={bgr[2]}")
    print(f"  HSV : H={h}  S={s}  V={v}")
    print(f"\n  Paste this into zed-color.py  target_hsv_ranges:")
    print(f"    (np.array([{h_lo}, {s_lo}, {v_lo}]), np.array([{h_hi}, 255, 255])),  # sampled H={h} S={s} V={v}")

    # Draw marker on the frozen/live frame
    display = state["frame"].copy()
    cv2.circle(display, (x, y), 10, (0, 255, 0), 2)
    cv2.putText(display, f"H={h} S={s} V={v}", (x + 12, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("ZED Shirt Color Finder", display)


def main():
    zed = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NONE   # no depth needed — faster
    init.camera_fps = 30

    print("Opening ZED camera...")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR opening ZED: {status}")
        sys.exit(1)
    print("ZED opened.  SPACE=freeze/unfreeze  s=save  click=sample  q/ESC=quit\n")

    image_zed = sl.Mat()
    runtime   = sl.RuntimeParameters()

    cv2.namedWindow("ZED Shirt Color Finder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ZED Shirt Color Finder", 1280, 720)
    cv2.setMouseCallback("ZED Shirt Color Finder", on_click)

    while True:
        if not state["frozen"]:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                bgra  = image_zed.get_data()
                frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
                state["frame"] = frame
                state["hsv"]   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if state["frame"] is not None:
            display = state["frame"].copy()
            label = "FROZEN — click shirt | SPACE=unfreeze" if state["frozen"] else "LIVE — press SPACE to freeze"
            cv2.putText(display, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("ZED Shirt Color Finder", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            state["frozen"] = not state["frozen"]
            print("Frame FROZEN — click to sample." if state["frozen"] else "Resuming live feed.")
        elif key == ord('s') and state["frame"] is not None:
            path = "shirt_snapshot.jpg"
            cv2.imwrite(path, state["frame"])
            print(f"Saved: {path}")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
