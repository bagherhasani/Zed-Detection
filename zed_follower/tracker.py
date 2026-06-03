import math
import time

import pyzed.sl as sl

from perception import get_torso_polygon, color_ratio


class TargetTracker:

    def __init__(self, config):
        self.cfg = config
        self.locked_id = None        # ID of the person 
        self.lock_lost_time = None   # when person lost in scenece
        self.last_seen_time = 0.0    # last seen time
        self.last_angle_error = 0.3  # last known direction

    def update(self, bodies, mask_color, img_w, img_h):
        now = time.time()

        for b in bodies.body_list:

            # skip unseccsful detections
            if b.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                continue

            # if locked, only look at the person we already chose
            if self.locked_id is not None and b.id != self.locked_id:
                continue

            # check how much of their shirt matches the target colour
            poly = get_torso_polygon(b.keypoint_2d, img_w, img_h)
            ratio = color_ratio(mask_color, poly)

            if ratio < self.cfg.color_ratio_threshold:
                continue  # not wearing the right colour — skip

            # found the target — update state
            self.locked_id = b.id
            self.lock_lost_time = None
            self.last_seen_time = now

            # get real-world distance and angle from ZED
            X = float(b.position[0])  # left / right
            Y = float(b.position[1])  # up / down
            Z = float(b.position[2])  # forward / backward
            distance    = math.sqrt(X**2 + Y**2 + Z**2)
            angle_error = -math.atan2(X, Z)
            self.last_angle_error = angle_error
            return b, poly, distance, angle_error

        # target not found this frame
        if self.locked_id is not None:
            if self.lock_lost_time is None:
                self.lock_lost_time = now  # start the lost timer
            elif (now - self.lock_lost_time) >= self.cfg.lock_lost_timeout_sec:
                self.locked_id = None      # give up register new person
                self.lock_lost_time = None

        return None

