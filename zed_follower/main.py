import time

import cv2
import pyzed.sl as sl
import rclpy
from rclpy.node import Node

from config import FollowerConfig
from perception import compute_color_mask
from tracker import TargetTracker
from controller import FollowController, ROSSender
from ui import draw_person_overlay


def setup_zed():
    # zed camera setup
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720   # quality low for faster fps
    init.depth_mode = sl.DEPTH_MODE.NEURAL          # depth accuracy
    init.coordinate_units = sl.UNIT.METER           # unit in metres

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("cannot open ZED camera !!!! chcek if connected and congfiged well")

    try:
        zed.enable_positional_tracking(sl.PositionalTrackingParameters())  # slam of zed camera
    except Exception:
        pass

    body_enabled = False
    try:
        bp = sl.BodyTrackingParameters() # stable id for each person
        bp.enable_tracking = True           # keep IDs stable across frames
        bp.enable_body_fitting = True       # fit a skeleton to each person
        bp.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
        bp.body_format = sl.BODY_FORMAT.BODY_38  # 38-keypoint skeleton (includes shoulders/hips)
        body_enabled = zed.enable_body_tracking(bp) == sl.ERROR_CODE.SUCCESS
    except Exception:
        pass

    return zed, body_enabled


def main():
    #  start ROS2 to send velcoty command to robot
    rclpy.init()
    node = Node("zed_follower")

    
    cfg = FollowerConfig() # load configs=
    tracker = TargetTracker(cfg)         # initialzie tracker with the configs (Who to follow)
    controller = FollowController(cfg, ROSSender(node))  # Initizlize speed (How fast to follow)

    # ZED camera realt prarams 
    zed, body_enabled = setup_zed()
    runtime_params = sl.RuntimeParameters()          
    bodies = sl.Bodies()                             
    body_rt = sl.BodyTrackingRuntimeParameters()
    body_rt.detection_confidence_threshold = 40     
    image = sl.Mat()           # empty container for image

    # main loop
    while True:

        # grab a new frame
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        # BGRA to BGR
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame_bgr = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        img_h, img_w = frame_bgr.shape[:2] # height and width of the image

        # white pixels = target colour, black = everything else
        mask_color = compute_color_mask(frame_bgr, cfg.target_hsv_lower, cfg.target_hsv_upper)

        # ask the tracker whether the target person is visible this frame
        target = None
        if body_enabled:
            zed.retrieve_bodies(bodies, body_rt)                    # run body detection
            target = tracker.update(bodies, mask_color, img_w, img_h)  # returns data or None

        if target:
            # unpack the target data
            body, poly, distance, angle_error = target

            # Follwo them 
            controller.follow(distance, angle_error)

            # draw their toros 
            draw_person_overlay(frame_bgr, body, poly)

            if distance:
                status = f"FOLLOWING  dist={distance:.2f}m"
            else:
                status = "FOLLOWING"

        else:
            # how many seconds since we last saw the target
            if tracker.last_seen_time > 0:
                since_seen = time.time() - tracker.last_seen_time
            else:
                since_seen = 999

            if cfg.search_enabled and since_seen < cfg.search_timeout_sec:
                controller.search(tracker.last_angle_error)  # rotate toward last known direction
                status = "SEARCHING"
            else:
                controller.stop()   # give up and wait
                status = "NO TARGET"

        # print status in top-left 
        cv2.putText(frame_bgr, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ZED Person Follow", frame_bgr)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):  # press q or ESC to quit
            break

    #  clean shutdown 
    controller.stop()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
