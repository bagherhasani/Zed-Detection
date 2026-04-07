import time
import math
import os

import cv2
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENGINE_PATH = os.path.join(_SCRIPT_DIR, "osnet_x025_reid.engine")

# ImageNet normalization (matches torchreid training)
_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OsNetReID:
    """Thin TensorRT wrapper for OSNet-x0.25 re-ID inference.

    Input : BGR crop (any size) → resized to 128×256, RGB, ImageNet-normalised
    Output: L2-normalised 512-dim embedding (np.float32)
    """

    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise RuntimeError("tensorrt / pycuda not installed")
        cuda.init()
        self._cuda_ctx = cuda.Device(0).make_context()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()

            # Allocate pinned host + device buffers once
            self._input_shape  = (1, 3, 256, 128)   # NCHW
            self._output_shape = (1, 512)
            nbytes_in  = int(np.prod(self._input_shape))  * 4   # float32
            nbytes_out = int(np.prod(self._output_shape)) * 4

            self._h_in  = cuda.pagelocked_empty(self._input_shape,  dtype=np.float32)
            self._h_out = cuda.pagelocked_empty(self._output_shape, dtype=np.float32)
            self._d_in  = cuda.mem_alloc(nbytes_in)
            self._d_out = cuda.mem_alloc(nbytes_out)
            self._stream = cuda.Stream()
        except Exception:
            self._cuda_ctx.pop()
            raise

    def __del__(self):
        try:
            self._cuda_ctx.pop()
            self._cuda_ctx.detach()
        except Exception:
            pass

    def _preprocess(self, frame_bgr: np.ndarray, bbox) -> np.ndarray:
        """Crop → 128×256 → RGB → float32 → normalise → NCHW."""
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x1 = int(max(0, x1));  y1 = int(max(0, y1))
        x2 = int(min(w, x2));  y2 = int(min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - _REID_MEAN) / _REID_STD
        return crop.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,256,128)

    def extract(self, frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        """Return 512-dim L2-normalised embedding, or None if crop is invalid."""
        inp = self._preprocess(frame_bgr, bbox)
        if inp is None:
            return None
        self._cuda_ctx.push()
        try:
            np.copyto(self._h_in, inp)
            cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
            self._context.execute_async_v2(
                bindings=[int(self._d_in), int(self._d_out)],
                stream_handle=self._stream.handle,
            )
            cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
            self._stream.synchronize()
            emb = self._h_out[0].copy()
        finally:
            self._cuda_ctx.pop()
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb

    @staticmethod
    def similarity(e1: np.ndarray, e2: np.ndarray) -> float:
        """Cosine similarity (both embeddings assumed L2-normalised)."""
        if e1 is None or e2 is None:
            return 0.0
        return float(np.dot(e1, e2))


class ZedFollower(Node):
    def __init__(self):
        super().__init__("zed_color_follower")

        # Publisher to Tracer base velocity command
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Actuation
        # If False, the robot will never move (status-only demo).
        self.actuation_enabled = True

        # Control parameters (very conservative defaults)
        # --- You can safely tweak these numbers ---
        self.target_distance = 1.0  # desired distance to target [m]
        self.min_distance = 0.6     # stop if closer than this [m]
        # Reduced speeds (safer/slower)
        self.max_lin_speed = 0.4   # maximum forward speed [m/s]
        self.max_ang_speed = 0.8   # maximum turn speed [rad/s]
        self.k_lin = 0.7            # linear gain  (smaller = slower accel)
        self.k_ang = 1.0            # angular gain (smaller = slower turn)
        


        # For on-screen feedback
        self.last_lin = 0.0          # last commanded linear speed
        self.last_ang = 0.0          # last commanded angular speed
        self.last_status = "Idle"    # human-readable robot state

        # Target memory (for safe search behavior)
        self.last_angle_error = 0.3      # sign indicates last known direction (+=left, -=right)
        self.last_seen_time = 0.0        # last time we saw the blob (even without depth)
        self.search_enabled = True       # rotate-in-place search when target is lost
        self.search_start_delay_sec = 1.0  # wait this long after losing target before rotating (avoid flicker)
        self.search_turn_speed = 0.5    # rad/s (in-place only)
        self.search_timeout_sec = 10.0   # rotate up to this long after last seen, then stop
        self.search_full_turn = True     # when target is lost, complete a full 360° scan
        self.search_angle_rad = 0.0      # integrated angle during current search
        self._search_prev_t = None       # internal timer for integration

        # Target locking (avoid switching between multiple people)
        self.locked_id = None
        self.lock_lost_time = None
        # Keep lock longer than the search window so we keep chasing the same person during search.
        self.lock_lost_timeout_sec = 12.0  # if locked target is missing for this long -> unlock
        self.last_target_position = None  # last known target position (x,y,z) in meters
        self.logical_target_id = None     # stable ID we show on screen (doesn't change if ZED id changes)
        # Re-ID policy: never switch to another same-color person.
        # If the locked person disappears, we will only reacquire near the last 3D position.
        # Otherwise we keep searching/stopped until timeout unlocks.
        self.never_switch_target = True
        self.relock_max_dist_m = 0.8       # only reacquire within this 3D distance of last target position
        self.relock_score_margin = 0.03    # if two candidates are too close, treat as ambiguous (do not switch)

        # OSNet re-ID (appearance-based identity, independent of ZED numeric IDs)
        self.target_embedding = None       # 512-dim L2-normalised np.float32; set at first lock
        self.reid_threshold = 0.60         # cosine-sim gate (tune 0.55-0.70 if locking is too strict/loose)
        self.reid_update_alpha = 0.05      # EMA rate for slowly updating stored embedding
        try:
            self.reid = OsNetReID(_ENGINE_PATH)
            self.get_logger().info("OSNet re-ID engine loaded.")
        except Exception as e:
            self.reid = None
            self.get_logger().warn(f"OSNet re-ID unavailable, falling back to color-only: {e}")

        # ZED related
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        # NOTE: Higher resolution = better image quality, but lower FPS on Jetson Nano.
        # Quality preset (sharper image):
        init_params.camera_resolution = sl.RESOLUTION.HD720
        # Higher-quality depth (can reduce FPS). If it becomes too slow, switch back to PERFORMANCE.
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        # Optional: cap FPS for stability (uncomment if needed)
        # init_params.camera_fps = 30
        init_params.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise RuntimeError("Cannot open ZED camera")

        # Prefer "human-first" mode:
        # detect PERSON with ZED object detection, then check for your target shirt color in torso ROI.
        self.person_mode = True
        self.color_ratio_threshold = 0.06  # 6% of torso ROI pixels must match target color
        self.enable_object_fallback = False  # ONLY follow people; never follow standalone objects
        self.objdet_enabled = False
        self.objects = sl.Objects()
        self.obj_runtime = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime.detection_confidence_threshold = 50

        # Enable positional tracking (helps object tracking; if it fails we still try detection)
        try:
            tracking_params = sl.PositionalTrackingParameters()
            self.zed.enable_positional_tracking(tracking_params)
        except Exception:
            pass

        # Enable ZED Body Tracking (skeleton) if available.
        # This lets us define the shirt region from keypoints (shoulders/hips) instead of a rectangle.
        self.body_enabled = False
        self.bodies = sl.Bodies()
        self.body_runtime = sl.BodyTrackingRuntimeParameters()
        self.body_runtime.detection_confidence_threshold = 40
        try:
            body_param = sl.BodyTrackingParameters()
            body_param.enable_tracking = True
            body_param.enable_body_fitting = True
            body_param.enable_segmentation = True
            body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
            body_param.body_format = sl.BODY_FORMAT.BODY_38
            res_bt = self.zed.enable_body_tracking(body_param)
            self.body_enabled = (res_bt == sl.ERROR_CODE.SUCCESS)
        except Exception:
            self.body_enabled = False

        try:
            obj_params = sl.ObjectDetectionParameters()
            obj_params.enable_tracking = True
            # Fast model is best for Jetson Nano
            if hasattr(sl, "OBJECT_DETECTION_MODEL") and hasattr(sl.OBJECT_DETECTION_MODEL, "MULTI_CLASS_BOX_FAST"):
                obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
            # Fallbacks (if enum differs)
            elif hasattr(sl, "OBJECT_DETECTION_MODEL") and hasattr(sl.OBJECT_DETECTION_MODEL, "MULTI_CLASS_BOX_MEDIUM"):
                obj_params.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
            res = self.zed.enable_object_detection(obj_params)
            self.objdet_enabled = (res == sl.ERROR_CODE.SUCCESS)
        except Exception:
            self.objdet_enabled = False

        # Target color (HSV ranges in OpenCV units):
        # H: 0..179, S: 0..255, V: 0..255
        #
        # Target shirt color (HSV ranges in OpenCV units).
        # OpenCV HSV: H 0..179, S 0..255, V 0..255.
        #
        # Shirt color calibrated from ZED camera (13 samples).
        # H=113–115 (center 114), S=163–203, V=73–171.
        self.target_hsv_ranges = [
            # Single tight range — covers bright and shadowed areas of shirt
            (np.array([108, 100, 40]), np.array([122, 255, 255])),
        ]

        # Shirt ROI policy:
        # - "FULL": shoulders->hips (covers whole t-shirt; more tolerant, but can catch undershirt strips)
        # - "UPPER": shoulders->mid-torso (more strict; better to avoid undershirt/waist false positives)
        self.shirt_roi_mode = "FULL"
        self.upper_torso_alpha = 0.55  # only used when shirt_roi_mode == "UPPER"

        # Shirt-shape filter (cheap anti-cheat): require color spread like a worn shirt
        # - must appear in BOTH left and right halves of ROI
        # - must cover minimum vertical span of ROI
        self.shirt_shape_filter = True
        self.shape_min_half_pixels = 180     # min colored pixels in each half
        self.shape_min_total_pixels = 700    # min colored pixels total
        self.shape_min_vert_span = 0.28      # fraction of ROI height spanned by colored pixels

        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.point_cloud = sl.Mat()

        # Kernel for smoothing the mask
        self.kernel = np.ones((5, 5), np.uint8)

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        self.get_logger().info(
            "ZED color follower started. Press 'q' or ESC in the window to quit."
        )

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        # Always safe to publish a stop (even in gesture-only mode)
        self.cmd_pub.publish(twist)
        self.last_lin = 0.0
        self.last_ang = 0.0
        self.last_status = "STOPPED"

    def search_for_target(self):
        """Rotate in place to reacquire target.

        If search_full_turn is True, we integrate angular speed and perform a full 360° scan
        (still safe: linear.x=0). If the target isn't found after a full turn, we keep rotating
        until search_timeout_sec expires (or you can change that behavior).
        """
        twist = Twist()
        twist.linear.x = 0.0  # safety: never drive forward while searching

        if not self.actuation_enabled:
            # Gesture-only demo: do not rotate the robot
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "SEARCHING (disabled)"
            return

        # Direction: positive angle_error => target was LEFT => turn left (positive z)
        direction = 1.0 if self.last_angle_error >= 0.0 else -1.0
        w = direction * min(self.search_turn_speed, self.max_ang_speed)

        # Track how much we've rotated during this search episode
        now = time.time()
        if self._search_prev_t is None:
            self._search_prev_t = now
        dt = max(0.0, now - self._search_prev_t)
        self._search_prev_t = now
        self.search_angle_rad += abs(w) * dt

        # Stop turning after one full 360° if requested
        if self.search_full_turn and self.search_angle_rad >= (2.0 * math.pi):
            twist.angular.z = 0.0
            self.last_status = "SEARCH DONE (360°) - STOPPED"
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.cmd_pub.publish(twist)
            return

        twist.angular.z = w

        self.last_lin = 0.0
        self.last_ang = float(twist.angular.z)
        self.last_status = "SEARCHING (rotate in place)"
        self.cmd_pub.publish(twist)

    def control_robot(self, distance, angle_rad):
        """
        Simple follow controller:
        - Keep target_distance from object
        - Turn to keep object centered
        """
        if not self.actuation_enabled:
            # Gesture-only demo: never move the robot
            self.last_lin = 0.0
            self.last_ang = 0.0
            self.last_status = "LOCKED (no actuation)"
            return

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

    def _pt(self, keypoints_2d, part):
        """Safe getter for a BODY_38_PARTS keypoint (returns (x,y) or None)."""
        idx = part.value
        if idx < 0 or idx >= len(keypoints_2d):
            return None
        x, y = keypoints_2d[idx]
        if x <= 0 or y <= 0:
            return None
        return float(x), float(y)

    def _bbox_from_4pts(self, bb2d):
        """Convert ZED 2D bbox (4 points) to x1,y1,x2,y2 integers."""
        pts = np.array([[p[0], p[1]] for p in bb2d], dtype=np.float32)
        x1 = int(np.clip(np.min(pts[:, 0]), 0, 10**9))
        y1 = int(np.clip(np.min(pts[:, 1]), 0, 10**9))
        x2 = int(np.clip(np.max(pts[:, 0]), 0, 10**9))
        y2 = int(np.clip(np.max(pts[:, 1]), 0, 10**9))
        return x1, y1, x2, y2

    def _torso_polygon_from_keypoints(self, keypoints_2d, img_w, img_h):
        """Build a torso polygon (shoulders+hips) from BODY_38 keypoints."""
        LS = sl.BODY_38_PARTS.LEFT_SHOULDER.value
        RS = sl.BODY_38_PARTS.RIGHT_SHOULDER.value
        LH = sl.BODY_38_PARTS.LEFT_HIP.value
        RH = sl.BODY_38_PARTS.RIGHT_HIP.value

        pts = []
        for idx in (LS, RS, RH, LH):
            x, y = keypoints_2d[idx]
            if x <= 0 or y <= 0:
                return None
            if x >= img_w or y >= img_h:
                return None
            pts.append([int(x), int(y)])

        # Slightly expand downward to cover more shirt area (helps if zipper opened).
        # Move hips down a bit (10% of shoulder-hip vertical distance)
        dy = int(0.10 * (max(pts[2][1], pts[3][1]) - min(pts[0][1], pts[1][1])))
        pts[2][1] = min(img_h - 1, pts[2][1] + dy)
        pts[3][1] = min(img_h - 1, pts[3][1] + dy)
        return np.array(pts, dtype=np.int32)

    def _upper_torso_polygon_from_keypoints(self, keypoints_2d, img_w, img_h, alpha=0.55):
        """Upper-torso (shirt) polygon: shoulders -> mid-torso (avoids waist/undershirt)."""
        LS = sl.BODY_38_PARTS.LEFT_SHOULDER.value
        RS = sl.BODY_38_PARTS.RIGHT_SHOULDER.value
        LH = sl.BODY_38_PARTS.LEFT_HIP.value
        RH = sl.BODY_38_PARTS.RIGHT_HIP.value

        # Read points
        try:
            lsx, lsy = keypoints_2d[LS]
            rsx, rsy = keypoints_2d[RS]
            lhx, lhy = keypoints_2d[LH]
            rhx, rhy = keypoints_2d[RH]
        except Exception:
            return None

        pts_raw = [(lsx, lsy), (rsx, rsy), (rhx, rhy), (lhx, lhy)]
        for x, y in pts_raw:
            if x <= 0 or y <= 0:
                return None
            if x >= img_w or y >= img_h:
                return None

        # Interpolate lower corners up from hips toward shoulders.
        # alpha=0 -> shoulders, alpha=1 -> hips. We pick ~0.55 (upper-mid torso).
        a = float(np.clip(alpha, 0.2, 0.9))
        llx = lsx + a * (lhx - lsx)
        lly = lsy + a * (lhy - lsy)
        rlx = rsx + a * (rhx - rsx)
        rly = rsy + a * (rhy - rsy)

        pts = [
            [int(lsx), int(lsy)],
            [int(rsx), int(rsy)],
            [int(rlx), int(rly)],
            [int(llx), int(lly)],
        ]
        return np.array(pts, dtype=np.int32)

    def _color_ratio_polygon_mask(self, mask_color, poly_pts):
        """Compute fraction of target-color pixels inside a polygon, using a precomputed color mask."""
        if poly_pts is None or len(poly_pts) < 3 or mask_color is None:
            return 0.0

        h_img, w_img = mask_color.shape[:2]
        x, y, w, h = cv2.boundingRect(poly_pts)
        if w <= 0 or h <= 0:
            return 0.0
        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Shift polygon into ROI coordinates
        poly_roi = (poly_pts - np.array([x1, y1], dtype=np.int32)).astype(np.int32)
        mask_poly = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask_poly, [poly_roi], 255)
        area = int(np.count_nonzero(mask_poly))
        if area <= 0:
            return 0.0

        mask_color_roi = mask_color[y1:y2, x1:x2]
        both = cv2.bitwise_and(mask_color_roi, mask_poly)
        return float(np.count_nonzero(both)) / float(area)

    def _color_ratio_polygon_mask_valid(self, mask_color, poly_pts, valid_mask):
        """Color ratio inside polygon, but only where valid_mask is nonzero (e.g., person segmentation)."""
        if poly_pts is None or len(poly_pts) < 3 or mask_color is None or valid_mask is None:
            return 0.0

        h_img, w_img = mask_color.shape[:2]
        if valid_mask.shape[:2] != (h_img, w_img):
            return 0.0

        x, y, w, h = cv2.boundingRect(poly_pts)
        if w <= 0 or h <= 0:
            return 0.0
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        poly_roi = (poly_pts - np.array([x1, y1], dtype=np.int32)).astype(np.int32)
        mask_poly = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask_poly, [poly_roi], 255)

        valid_roi = valid_mask[y1:y2, x1:x2]
        denom = cv2.bitwise_and(valid_roi, mask_poly)
        denom_area = int(np.count_nonzero(denom))
        if denom_area <= 0:
            return 0.0

        mask_color_roi = mask_color[y1:y2, x1:x2]
        num = cv2.bitwise_and(mask_color_roi, denom)
        return float(np.count_nonzero(num)) / float(denom_area)

    def _shirt_color_stats(self, mask_color, poly_pts, valid_mask=None):
        """Compute shirt-color stats inside polygon (optionally gated by valid_mask).

        Returns (ratio, total_colored, left_colored, right_colored, vert_span_frac, denom_area).
        """
        if poly_pts is None or len(poly_pts) < 3 or mask_color is None:
            return 0.0, 0, 0, 0, 0.0, 0

        h_img, w_img = mask_color.shape[:2]
        x, y, w, h = cv2.boundingRect(poly_pts)
        if w <= 0 or h <= 0:
            return 0.0, 0, 0, 0, 0.0, 0
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0, 0, 0, 0.0, 0

        poly_roi = (poly_pts - np.array([x1, y1], dtype=np.int32)).astype(np.int32)
        mask_poly = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(mask_poly, [poly_roi], 255)

        denom = mask_poly
        if valid_mask is not None and valid_mask.shape[:2] == (h_img, w_img):
            valid_roi = valid_mask[y1:y2, x1:x2]
            denom = cv2.bitwise_and(valid_roi, mask_poly)

        denom_area = int(np.count_nonzero(denom))
        if denom_area <= 0:
            return 0.0, 0, 0, 0, 0.0, 0

        mask_color_roi = mask_color[y1:y2, x1:x2]
        num = cv2.bitwise_and(mask_color_roi, denom)

        total_col = int(np.count_nonzero(num))
        ratio = float(total_col) / float(denom_area)

        # Left/right halves coverage
        mid = (x2 - x1) // 2
        left_col = int(np.count_nonzero(num[:, :mid])) if mid > 0 else 0
        right_col = int(np.count_nonzero(num[:, mid:])) if mid < (x2 - x1) else 0

        # Vertical span coverage
        ys, xs = np.where(num > 0)
        if ys.size == 0:
            span_frac = 0.0
        else:
            span = int(ys.max() - ys.min() + 1)
            span_frac = float(span) / float(max(1, (y2 - y1)))

        return ratio, total_col, left_col, right_col, span_frac, denom_area

    def _color_ratio_polygon(self, frame_bgr, poly_pts):
        """Compute fraction of target-color pixels inside a polygon mask."""
        if poly_pts is None or len(poly_pts) < 3:
            return 0.0
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask_color = None
        for lower, upper in self.target_hsv_ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask_color = m if mask_color is None else cv2.bitwise_or(mask_color, m)
        return self._color_ratio_polygon_mask(mask_color, poly_pts)

    def _color_ratio(self, frame_bgr, x1, y1, x2, y2):
        """Compute fraction of target-color pixels in ROI."""
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = None
        for lower, upper in self.target_hsv_ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return float(np.count_nonzero(mask)) / float(mask.size)


# ═══════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ui_rect(img, x1, y1, x2, y2, color_bgr, alpha=0.60):
    """Semi-transparent filled rectangle."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color_bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _ui_text(img, text, x, y, scale=0.60, color=(255, 255, 255), thickness=1):
    """Text with drop-shadow for readability on any background."""
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _ui_tsize(text, scale=0.60, thickness=1):
    (w, h), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h + bl


def _corner_box(img, x1, y1, x2, y2, color, thickness=2, ratio=0.22):
    """Corner-bracket bounding box (no full rectangle — cleaner look)."""
    cw = max(12, int((x2 - x1) * ratio))
    ch = max(12, int((y2 - y1) * ratio))
    corners = [
        [(x1, y1 + ch), (x1, y1),       (x1 + cw, y1)],
        [(x2 - cw, y1), (x2, y1),       (x2, y1 + ch)],
        [(x2, y2 - ch), (x2, y2),       (x2 - cw, y2)],
        [(x1 + cw, y2), (x1, y2),       (x1, y2 - ch)],
    ]
    for p0, apex, p1 in corners:
        cv2.line(img, p0, apex, color, thickness, cv2.LINE_AA)
        cv2.line(img, apex, p1, color, thickness, cv2.LINE_AA)


def _badge(img, text, cx, cy, fg=(255, 255, 255), bg=(30, 30, 30),
           border=(0, 220, 255), scale=0.72, thickness=2, pad_x=14, pad_y=8):
    """Pill-shaped label badge centered at (cx, cy)."""
    tw, th = _ui_tsize(text, scale, thickness)
    x1 = cx - tw // 2 - pad_x
    y1 = cy - th // 2 - pad_y
    x2 = cx + tw // 2 + pad_x
    y2 = cy + th // 2 + pad_y
    _ui_rect(img, x1, y1, x2, y2, bg, alpha=0.75)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
    tx = cx - tw // 2
    ty = cy + th // 2 - 2
    _ui_text(img, text, tx, ty, scale, fg, thickness)


def draw_person_overlay(img, best_body, best_poly, logical_id,
                        color_ratio, reid_sim, reid_enabled,
                        dist_text, _pt_fn):
    """Draw corner box, torso polygon and ID badge over the detected person."""
    h, w = img.shape[:2]

    # Corner box from bounding box
    bb = best_body.bounding_box_2d
    bx1 = int(max(0, min(p[0] for p in bb)))
    by1 = int(max(0, min(p[1] for p in bb)))
    bx2 = int(min(w, max(p[0] for p in bb)))
    by2 = int(min(h, max(p[1] for p in bb)))
    _corner_box(img, bx1, by1, bx2, by2, (0, 220, 255), thickness=3)

    # Torso polygon (semi-transparent fill + outline)
    overlay = img.copy()
    cv2.fillPoly(overlay, [best_poly], (255, 140, 0))
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [best_poly.reshape(-1, 1, 2)], True, (0, 200, 255), 2, cv2.LINE_AA)

    # ID badge above head
    neck = _pt_fn(best_body.keypoint_2d, sl.BODY_38_PARTS.NECK)
    if neck is not None:
        nx, ny = int(neck[0]), int(neck[1])
        badge_cy = max(30, ny - 32)
        _badge(img, f"TARGET  #{logical_id}", nx, badge_cy,
               fg=(255, 255, 255), bg=(10, 10, 40),
               border=(0, 220, 255), scale=0.78, thickness=2)

    # Small info strip just below torso polygon bottom-left corner
    info_x = bx1 + 6
    info_y = by2 + 18
    if info_y + 20 < h:
        col_str = f"color {color_ratio*100:.0f}%"
        reid_str = f"  re-ID {reid_sim:.2f}" if reid_enabled else ""
        info_str = col_str + reid_str
        tw, th = _ui_tsize(info_str, 0.48)
        _ui_rect(img, info_x - 4, info_y - th - 2, info_x + tw + 4, info_y + 4,
                 (10, 10, 10), alpha=0.60)
        _ui_text(img, info_str, info_x, info_y, 0.48,
                 color=(180, 255, 180) if color_ratio >= 0.10 else (180, 180, 180))


def draw_hud(img, fps, detected, status_str, lin_speed, ang_speed,
             logical_id, dist_text, color_ratio, reid_sim, reid_enabled):
    """Full telemetry HUD: top bar, left panel, bottom bar, no-target banner."""
    h, w = img.shape[:2]
    AA = cv2.LINE_AA

    # ── TOP BAR ─────────────────────────────────────────────────────────────
    top_h = 54
    _ui_rect(img, 0, 0, w, top_h, (8, 8, 8), alpha=0.70)
    cv2.line(img, (0, top_h), (w, top_h), (0, 180, 220), 1, AA)
    _ui_text(img, "ZED 2i  |  PERSON TRACKER", 16, 36,
             scale=0.90, color=(0, 220, 255), thickness=2)
    fps_str = f"{fps:.1f} FPS"
    fps_col = (0, 255, 120) if fps >= 20 else (0, 200, 255) if fps >= 12 else (60, 60, 255)
    tw, _ = _ui_tsize(fps_str, 0.72, 2)
    _ui_text(img, fps_str, w - tw - 16, 36, scale=0.72, color=fps_col, thickness=2)

    # ── LEFT PANEL ──────────────────────────────────────────────────────────
    px, py = 14, top_h + 12
    pw, panel_inner_h = 298, 200
    _ui_rect(img, px, py, px + pw, py + panel_inner_h, (8, 8, 8), alpha=0.65)
    cv2.rectangle(img, (px, py), (px + pw, py + panel_inner_h), (50, 50, 60), 1, AA)
    cv2.line(img, (px, py + 1), (px + pw, py + 1), (0, 160, 200), 2, AA)

    # Status dot + label
    if detected:
        dot_col, state_col = (0, 255, 100), (0, 255, 130)
    elif "SEARCH" in status_str.upper():
        dot_col, state_col = (0, 200, 255), (0, 200, 255)
    else:
        dot_col, state_col = (60, 60, 220), (120, 120, 220)
    cv2.circle(img, (px + 18, py + 24), 8, dot_col, -1, AA)
    cv2.circle(img, (px + 18, py + 24), 8, (200, 200, 200), 1, AA)
    _ui_text(img, status_str[:32], px + 34, py + 30, scale=0.58,
             color=state_col, thickness=1)

    cv2.line(img, (px + 10, py + 44), (px + pw - 10, py + 44), (50, 50, 60), 1)

    # Target ID
    if logical_id is not None:
        tid_str = f"TARGET  #{logical_id}"
        tid_col = (0, 220, 255)
    else:
        tid_str = "NO  TARGET  LOCKED"
        tid_col = (100, 100, 130)
    _ui_text(img, tid_str, px + 12, py + 68, scale=0.70, color=tid_col, thickness=2)

    # Distance / angle
    _ui_text(img, dist_text if dist_text else "-- awaiting depth --",
             px + 12, py + 96, scale=0.55, color=(210, 210, 210))

    # Velocity row
    spd_str = f"v  {lin_speed:+.2f} m/s     w  {ang_speed:+.2f} rad/s"
    _ui_text(img, spd_str, px + 12, py + 120, scale=0.50, color=(160, 210, 255))

    # re-ID bar
    if reid_enabled:
        bar_label = "re-ID"
        bar_lx = px + 12
        bar_y  = py + 148
        bar_x0 = bar_lx + 46
        bar_x1 = px + pw - 12
        bar_bar_h = 12
        _ui_text(img, bar_label, bar_lx, bar_y, scale=0.45, color=(150, 150, 160))
        cv2.rectangle(img, (bar_x0, bar_y - bar_bar_h),
                      (bar_x1, bar_y + 1), (40, 40, 40), -1)
        filled = int((bar_x1 - bar_x0) * min(1.0, max(0.0, reid_sim)))
        bar_col = ((0, 255, 110) if reid_sim >= 0.70
                   else (0, 200, 255) if reid_sim >= 0.55
                   else (60, 60, 220))
        if filled > 0:
            cv2.rectangle(img, (bar_x0, bar_y - bar_bar_h),
                          (bar_x0 + filled, bar_y + 1), bar_col, -1)
        cv2.rectangle(img, (bar_x0, bar_y - bar_bar_h),
                      (bar_x1, bar_y + 1), (80, 80, 80), 1)
        sim_str = f"{reid_sim:.2f}"
        _ui_text(img, sim_str, bar_x1 + 4, bar_y, scale=0.45, color=bar_col)

    # Color match bar
    if detected and color_ratio > 0:
        clabel = "color"
        cl_lx  = px + 12
        cl_y   = py + 176
        cl_x0  = cl_lx + 46
        cl_x1  = px + pw - 12
        cl_h   = 10
        _ui_text(img, clabel, cl_lx, cl_y, scale=0.45, color=(150, 150, 160))
        cv2.rectangle(img, (cl_x0, cl_y - cl_h), (cl_x1, cl_y + 1), (40, 40, 40), -1)
        cfilled = int((cl_x1 - cl_x0) * min(1.0, color_ratio * 4))  # scale: 25% = full
        ccol = (80, 220, 80) if color_ratio >= 0.12 else (80, 180, 160)
        if cfilled > 0:
            cv2.rectangle(img, (cl_x0, cl_y - cl_h),
                          (cl_x0 + cfilled, cl_y + 1), ccol, -1)
        cv2.rectangle(img, (cl_x0, cl_y - cl_h), (cl_x1, cl_y + 1), (80, 80, 80), 1)
        _ui_text(img, f"{color_ratio*100:.0f}%", cl_x1 + 4, cl_y, scale=0.45, color=ccol)

    # ── BOTTOM BAR ──────────────────────────────────────────────────────────
    bot_y = h - 34
    _ui_rect(img, 0, bot_y, w, h, (8, 8, 8), alpha=0.65)
    cv2.line(img, (0, bot_y), (w, bot_y), (0, 100, 140), 1, AA)
    _ui_text(img, "[Q / ESC]  STOP", 16, h - 9, scale=0.52, color=(100, 130, 220))

    # ── NO-TARGET BANNER ────────────────────────────────────────────────────
    if not detected:
        if "360" in status_str:
            msg = "360\u00b0 SCAN COMPLETE  \u2014  NO TARGET"
        elif "SEARCH" in status_str.upper():
            msg = "\u25b6  SEARCHING  \u2014  ROTATING  \u25c0"
        else:
            msg = "SCANNING FOR TARGET PERSON..."
        tw, th = _ui_tsize(msg, scale=0.88, thickness=2)
        bx = (w - tw) // 2 - 16
        by = h // 2 - th - 18
        _ui_rect(img, bx, by, bx + tw + 32, by + th + 28, (0, 0, 140), alpha=0.60)
        cv2.rectangle(img, (bx, by), (bx + tw + 32, by + th + 28), (0, 80, 255), 2, AA)
        cv2.rectangle(img, (bx + 2, by + 2),
                      (bx + tw + 30, by + th + 26), (0, 40, 120), 1, AA)
        _ui_text(img, msg, bx + 16, by + th + 12,
                 scale=0.88, color=(255, 255, 255), thickness=2)


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
    last_frame_bgr = None
    hud_reid_sim    = 0.0   # persists across frames for smooth bar display
    hud_color_ratio = 0.0
    hud_dist_text   = ""

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()  # BGRA from ZED SDK
            # Convert BGRA -> BGR for OpenCV (otherwise red/blue are swapped)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            last_frame_bgr = frame_bgr

            img_h, img_w, _ = frame_bgr.shape
            detected = False
            dist_text = ""
            tag_text = ""
            target_cx = None
            target_cy = None

            # Precompute HSV + target color mask ONCE per frame (big FPS win).
            hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask_color_frame = None
            for lower, upper in node.target_hsv_ranges:
                m = cv2.inRange(hsv_frame, lower, upper)
                mask_color_frame = m if mask_color_frame is None else cv2.bitwise_or(mask_color_frame, m)

            # --- Mode 1: HUMAN-FIRST with SKELETON (Body Tracking + shirt polygon) ---
            if node.person_mode and node.body_enabled:
                zed.retrieve_bodies(node.bodies, node.body_runtime)

                best_body = None
                best_score = -1.0
                best_color_ratio = 0.0
                best_poly = None

                # Candidates:
                # - If locked_id is present in current frame, evaluate only that body (stable tracking)
                # - If locked_id is missing, NEVER switch to another person:
                #   we only consider bodies that are very close to the last known 3D position.
                candidates = node.bodies.body_list
                locked_present = False
                if node.locked_id is not None:
                    # Prefer exact locked numeric ID if still tracked
                    for b in node.bodies.body_list:
                        if b.id == node.locked_id:
                            candidates = [b]
                            locked_present = True
                            break

                    # If numeric ID disappeared, only consider very-close bodies (prevents switching)
                    if (not locked_present) and node.never_switch_target:
                        if node.last_target_position is None:
                            candidates = []
                        else:
                            tx, ty, tz = node.last_target_position
                            near = []
                            for b in node.bodies.body_list:
                                try:
                                    bx, by, bz = float(b.position[0]), float(b.position[1]), float(b.position[2])
                                    dist = math.sqrt((bx - tx) ** 2 + (by - ty) ** 2 + (bz - tz) ** 2)
                                    if dist <= node.relock_max_dist_m:
                                        near.append(b)
                                except Exception:
                                    pass
                            candidates = near

                scored = []
                for b in candidates:
                    if b.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                        continue
                    if len(b.keypoint_2d) == 0:
                        continue
                    if node.shirt_roi_mode == "UPPER":
                        poly = node._upper_torso_polygon_from_keypoints(
                            b.keypoint_2d, img_w, img_h, alpha=node.upper_torso_alpha
                        )
                    else:
                        # FULL shirt/torso ROI (shoulders->hips)
                        poly = node._torso_polygon_from_keypoints(b.keypoint_2d, img_w, img_h)
                    # If segmentation is available, only count pixels that belong to the person.
                    person_mask = None
                    try:
                        m = b.mask
                        if m is not None:
                            md = m.get_data()
                            if md is not None:
                                if md.ndim == 3:
                                    md = md[:, :, 0]
                                pm = (md > 0).astype(np.uint8) * 255
                                if pm.shape[0] > 0 and pm.shape[1] > 0:
                                    if pm.shape[:2] != (img_h, img_w):
                                        pm = cv2.resize(pm, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                                    person_mask = pm
                    except Exception:
                        person_mask = None

                    ratio, total_col, left_col, right_col, span_frac, _ = node._shirt_color_stats(
                        mask_color_frame, poly, valid_mask=person_mask
                    )

                    # Optional "shirt-like distribution" filter (anti-cheat vs held cloth blob)
                    if node.shirt_shape_filter:
                        if (
                            total_col < node.shape_min_total_pixels
                            or left_col < node.shape_min_half_pixels
                            or right_col < node.shape_min_half_pixels
                            or span_frac < node.shape_min_vert_span
                        ):
                            # Still keep ratio for overlay/debug, but don't accept as target
                            continue

                    # In "relock" mode (locked id missing), require color threshold and prefer closest to last 3D pos.
                    dist = 0.0
                    if (not locked_present) and (node.locked_id is not None) and (node.last_target_position is not None):
                        if ratio < node.color_ratio_threshold:
                            continue
                        try:
                            bx, by, bz = float(b.position[0]), float(b.position[1]), float(b.position[2])
                            tx, ty, tz = node.last_target_position
                            dist = math.sqrt((bx - tx) ** 2 + (by - ty) ** 2 + (bz - tz) ** 2)
                        except Exception:
                            dist = 999.0
                        score = ratio - 0.05 * dist  # keep mostly color-based, slight position preference
                    else:
                        score = ratio

                    # --- OSNet re-ID gate: skip bodies that don't look like the locked target ---
                    reid_sim = 0.0
                    if node.reid is not None and node.target_embedding is not None:
                        bbox = node._bbox_from_4pts(b.bounding_box_2d)
                        emb = node.reid.extract(frame_bgr, bbox)
                        reid_sim = OsNetReID.similarity(node.target_embedding, emb)
                        if reid_sim < node.reid_threshold:
                            continue  # appearance mismatch: not our target person
                        # Blend appearance into score (50/50 with existing color/position score)
                        score = 0.5 * score + 0.5 * reid_sim

                    scored.append((score, ratio, reid_sim, dist, b, poly))

                # Pick best candidate; if ambiguous in relock mode, pick none (never switch).
                if len(scored) > 0:
                    scored.sort(key=lambda t: t[0], reverse=True)
                    if (not locked_present) and (node.locked_id is not None) and len(scored) >= 2:
                        if (scored[0][0] - scored[1][0]) < node.relock_score_margin:
                            scored = []

                if len(scored) > 0:
                    best_score, best_color_ratio, best_reid_sim, _, best_body, best_poly = scored[0]

                if best_body is not None and best_color_ratio >= node.color_ratio_threshold:
                    # lock on this id
                    # Keep a stable logical id for display, even if ZED changes numeric IDs after occlusion
                    if node.logical_target_id is None:
                        node.logical_target_id = 1
                    node.locked_id = best_body.id
                    node.lock_lost_time = None

                    # Update OSNet embedding (EMA so it slowly adapts to lighting/pose changes)
                    if node.reid is not None:
                        bbox = node._bbox_from_4pts(best_body.bounding_box_2d)
                        new_emb = node.reid.extract(frame_bgr, bbox)
                        if new_emb is not None:
                            if node.target_embedding is None:
                                node.target_embedding = new_emb          # first lock (already L2-norm)
                            else:
                                blended = (
                                    (1.0 - node.reid_update_alpha) * node.target_embedding
                                    + node.reid_update_alpha * new_emb
                                )
                                # Re-normalize so dot-product stays cosine similarity
                                n = np.linalg.norm(blended)
                                if n > 1e-6:
                                    blended /= n
                                node.target_embedding = blended
                    # Update last known target position for re-locking after occlusion
                    try:
                        node.last_target_position = (
                            float(best_body.position[0]),
                            float(best_body.position[1]),
                            float(best_body.position[2]),
                        )
                    except Exception:
                        pass

                    # Reset search integrator because we have a target again
                    node.search_angle_rad = 0.0
                    node._search_prev_t = None

                    # Target pixel: polygon centroid
                    target_cx = int(np.mean(best_poly[:, 0]))
                    target_cy = int(np.mean(best_poly[:, 1]))

                    node.last_seen_time = time.time()
                    node.last_status = f"LOCKED id={node.locked_id} | FOLLOW"

                    # Distance/angle from BODY tracking 3D position (FAST, avoids point cloud)
                    # ZED body position is already in meters (camera frame): X=left/right, Y=up/down, Z=forward.
                    try:
                        X = float(best_body.position[0])
                        Y = float(best_body.position[1])
                        Z = float(best_body.position[2])
                        if (not math.isfinite(X)) or (not math.isfinite(Y)) or (not math.isfinite(Z)) or (Z <= 0.05):
                            raise ValueError("invalid body position")

                        distance = float(math.sqrt(X * X + Y * Y + Z * Z))
                        angle_raw = math.atan2(X, Z)
                        angle_error = -angle_raw  # positive => left
                        angle_err_deg = math.degrees(angle_error)

                        dir_text = "center"
                        if angle_err_deg > 3:
                            dir_text = "left"
                        elif angle_err_deg < -3:
                            dir_text = "right"

                        dist_text = f"{distance:.2f} m, {abs(angle_err_deg):.1f} deg {dir_text}"
                        node.last_angle_error = float(angle_error)  # for lost-target search direction
                        node.control_robot(distance, angle_error)
                    except Exception:
                        # Fallback: if body position is unavailable, steer using pixel offset only.
                        pixel_error = (target_cx - (img_w // 2)) / float(img_w)
                        angle_error = -pixel_error
                        node.last_angle_error = float(angle_error)
                        node.control_robot(None, angle_error)

                    # Person overlay (corner box + polygon + badge)
                    draw_person_overlay(
                        frame_bgr, best_body, best_poly,
                        node.logical_target_id,
                        best_color_ratio, best_reid_sim,
                        node.reid is not None,
                        dist_text, node._pt,
                    )
                    # Update persistent HUD state
                    hud_reid_sim    = best_reid_sim
                    hud_color_ratio = best_color_ratio
                    hud_dist_text   = dist_text
                    detected = True
                else:
                    # Locked target missing or not matching -> unlock after timeout
                    if node.locked_id is not None:
                        if node.lock_lost_time is None:
                            node.lock_lost_time = time.time()
                        elif (time.time() - node.lock_lost_time) >= node.lock_lost_timeout_sec:
                            node.locked_id = None
                            node.lock_lost_time = None
                            node.last_target_position = None
                            node.logical_target_id = None
                            node.target_embedding = None  # clear OSNet fingerprint on full unlock

            # --- Mode 2: optional fallback (DISABLED by default) ---
            # If you ever want to allow following standalone objects matching the target color,
            # set node.enable_object_fallback = True and re-add object tracking here.
            if not detected and node.enable_object_fallback:
                pass

            if not detected:
                # No target: after a short grace period, rotate in place to reacquire.
                now = time.time()
                since_seen = (now - node.last_seen_time) if node.last_seen_time > 0 else 1e9

                do_search = (
                    node.search_enabled
                    and node.last_seen_time > 0.0
                    and since_seen >= node.search_start_delay_sec
                )

                if do_search:
                    # If search_full_turn is enabled, ALWAYS complete one full 360° scan once started,
                    # regardless of timeout (safer + deterministic behavior).
                    if node.search_full_turn:
                        if node.search_angle_rad < (2.0 * math.pi):
                            node.search_for_target()  # uses node.last_angle_error for direction, linear.x=0
                        else:
                            node.stop_robot()
                            node.last_status = "NO TARGET (360° scan done)"
                    else:
                        # Timeout-based search (legacy option)
                        if since_seen <= node.search_timeout_sec:
                            node.search_for_target()
                        else:
                            node.stop_robot()
                            node.last_status = "NO TARGET (search timeout)"
                else:
                    node.stop_robot()
                    if node.last_seen_time > 0.0 and since_seen < node.search_start_delay_sec:
                        node.last_status = "NO TARGET (grace - stopped)"
                    else:
                        node.last_status = "NO TARGET (stopped)"

            # FPS estimate
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0

            # ── Unified HUD overlay ──────────────────────────────────────────
            draw_hud(
                frame_bgr,
                fps        = fps,
                detected   = detected,
                status_str = node.last_status,
                lin_speed  = node.last_lin,
                ang_speed  = node.last_ang,
                logical_id = node.logical_target_id,
                dist_text  = hud_dist_text if detected else "",
                color_ratio= hud_color_ratio if detected else 0.0,
                reid_sim   = hud_reid_sim   if detected else 0.0,
                reid_enabled = node.reid is not None,
            )

            cv2.imshow("ZED Person Follow (Color)", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
        else:
            # Grab failed briefly (USB hiccup / load). Show last frame to avoid "flash" / stutter.
            if last_frame_bgr is not None:
                frame_bgr = last_frame_bgr.copy()
                h_f, w_f = frame_bgr.shape[:2]
                _ui_rect(frame_bgr, 0, 0, w_f, 50, (0, 0, 140), alpha=0.70)
                _ui_text(frame_bgr, "ZED GRAB FAILED  -  SKIPPING FRAME",
                         14, 34, scale=0.75, color=(80, 80, 255), thickness=2)
                cv2.imshow("ZED Person Follow (Color)", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    # On exit: stop robot and clean up
    node.stop_robot()
    zed.close()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()