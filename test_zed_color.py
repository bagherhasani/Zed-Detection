"""
pytest test suite for ZedFollower pure-logic methods.

Covers:
  - control_robot(distance, angle_rad)
  - stop_robot()
  - search_for_target()
  - _shirt_color_ratio(mask_color_frame, body, img_h, img_w)

Run with:
  cd "/home/user/ros2_ws/src/zed-detection " && python -m pytest test_zed_color.py -v
"""

import math
import sys
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Patch sys.modules BEFORE importing the module under test.
# The hyphen in the filename prevents normal import; we use importlib.util.
# ---------------------------------------------------------------------------

# --- pyzed.sl mock ---
sl_mock = MagicMock()
# Make ERROR_CODE.SUCCESS compare equal to whatever .open() returns so that
# ZedFollower.__init__ passes the `if status != sl.ERROR_CODE.SUCCESS` guard.
sl_mock.ERROR_CODE.SUCCESS = "SUCCESS"

_real_open = sl_mock.Camera.return_value.open
_real_open.return_value = "SUCCESS"  # matches SUCCESS sentinel above

sys.modules["pyzed"] = MagicMock()
sys.modules["pyzed.sl"] = sl_mock

# --- rclpy mocks ---
sys.modules["rclpy"] = MagicMock()
sys.modules["rclpy.node"] = MagicMock()
# ZedFollower inherits from Node; use plain object so __new__ works cleanly.
sys.modules["rclpy.node"].Node = object

# --- geometry_msgs mock ---
# Twist must produce real attribute-bearing objects so the method logic works.


class MockVector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class MockTwist:
    def __init__(self):
        self.linear = MockVector3()
        self.angular = MockVector3()


geometry_msgs_mock = MagicMock()
geometry_msgs_mock.msg.Twist = MockTwist
sys.modules["geometry_msgs"] = geometry_msgs_mock
sys.modules["geometry_msgs.msg"] = geometry_msgs_mock.msg

# --- cv2 mock (avoid needing a display) ---
cv2_mock = MagicMock()
# Make bitwise_and return a zero numpy array by default (overridden per-test)
cv2_mock.bitwise_and.side_effect = lambda a, b: np.zeros_like(a)
cv2_mock.resize.side_effect = lambda src, dsize, **kw: src
sys.modules["cv2"] = cv2_mock

# --- Now import the module via importlib ---
import importlib.util

_MODULE_PATH = "/home/user/ros2_ws/src/zed-detection /zed-color.py"
spec = importlib.util.spec_from_file_location("zed_color", _MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ZedFollower = module.ZedFollower


# ---------------------------------------------------------------------------
# Fixture: create a ZedFollower instance bypassing __init__
# ---------------------------------------------------------------------------


def make_node(**overrides):
    """Create a ZedFollower without opening hardware, ready for unit testing."""
    node = ZedFollower.__new__(ZedFollower)

    # Attributes referenced by the four methods under test
    node.cmd_pub = MagicMock()
    node.actuation_enabled = True
    node.target_distance = 1.0
    node.min_distance = 0.6
    node.max_lin_speed = 0.2
    node.max_ang_speed = 0.5
    node.k_lin = 0.7
    node.k_ang = 1.0
    node.smooth_alpha = 0.35
    node.prev_lin = 0.0
    node.prev_ang = 0.0
    node.ang_deadzone_rad = 0.052
    node.last_lin = 0.0
    node.last_ang = 0.0
    node.last_status = "Idle"
    node.search_turn_speed = 0.25
    node.search_full_turn = True
    node.search_angle_rad = 0.0
    node._search_prev_t = None
    node.last_angle_error = 0.3
    node.target_hsv_ranges = [
        (np.array([100, 100, 50]), np.array([130, 255, 255])),
    ]
    # get_logger() is used inside _shirt_color_ratio; return a silent mock
    node.get_logger = MagicMock(return_value=MagicMock())

    for key, val in overrides.items():
        setattr(node, key, val)

    return node


@pytest.fixture
def node():
    return make_node()


# ===========================================================================
# stop_robot()
# ===========================================================================


class TestStopRobot:
    def test_publishes_zero_twist(self, node):
        node.stop_robot()
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.linear.x == 0.0
        assert twist.angular.z == 0.0

    def test_resets_prev_lin_to_zero(self, node):
        node.prev_lin = 0.18
        node.stop_robot()
        assert node.prev_lin == 0.0

    def test_resets_prev_ang_to_zero(self, node):
        node.prev_ang = -0.4
        node.stop_robot()
        assert node.prev_ang == 0.0

    def test_resets_last_lin_to_zero(self, node):
        node.last_lin = 0.15
        node.stop_robot()
        assert node.last_lin == 0.0

    def test_resets_last_ang_to_zero(self, node):
        node.last_ang = 0.3
        node.stop_robot()
        assert node.last_ang == 0.0

    def test_sets_last_status_to_stopped(self, node):
        node.stop_robot()
        assert node.last_status == "STOPPED"


# ===========================================================================
# control_robot(distance, angle_rad)
# ===========================================================================


class TestControlRobotActuationDisabled:
    def test_publishes_zero_twist_when_disabled(self, node):
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.linear.x == 0.0
        assert twist.angular.z == 0.0

    def test_returns_early_without_setting_movement_when_disabled(self, node):
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        assert node.last_lin == 0.0
        assert node.last_ang == 0.0

    def test_sets_disabled_status(self, node):
        node.actuation_enabled = False
        node.control_robot(2.0, 0.5)
        assert node.last_status == "LOCKED (no actuation)"


class TestControlRobotNeverReverses:
    def test_raw_linear_never_negative_when_distance_equals_target(self, node):
        """At exactly target_distance, error_d = 0, raw_lin = 0."""
        node.control_robot(node.target_distance, 0.0)
        assert node.last_lin >= 0.0

    def test_raw_linear_never_negative_when_distance_slightly_below_target(self, node):
        """distance just below target → error_d < 0 → clamped to 0, no reverse."""
        node.prev_lin = 0.0
        node.control_robot(0.9, 0.0)
        assert node.last_lin >= 0.0

    def test_no_reverse_when_distance_equals_min_distance(self, node):
        node.control_robot(node.min_distance, 0.0)
        assert node.last_lin >= 0.0

    def test_no_reverse_when_distance_just_above_min(self, node):
        node.control_robot(node.min_distance + 0.01, 0.0)
        assert node.last_lin >= 0.0


class TestControlRobotTooClose:
    def test_zero_raw_lin_when_distance_below_min(self, node):
        """distance < min_distance → raw_lin = 0; smoothing means last_lin trends to 0."""
        node.prev_lin = 0.0  # start from zero so smooth output is also zero
        node.control_robot(0.3, 0.0)
        assert node.last_lin == 0.0

    def test_status_too_close_when_distance_below_min(self, node):
        node.control_robot(0.3, 0.0)
        assert node.last_status == "TOO CLOSE - HOLDING"

    def test_zero_raw_lin_when_distance_is_none(self, node):
        node.prev_lin = 0.0
        node.control_robot(None, 0.0)
        assert node.last_lin == 0.0

    def test_status_too_close_when_distance_is_none(self, node):
        node.control_robot(None, 0.0)
        assert node.last_status == "TOO CLOSE - HOLDING"


class TestControlRobotAngularDeadzone:
    def test_no_angular_output_when_angle_exactly_zero(self, node):
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.0)
        assert node.last_ang == 0.0

    def test_no_angular_output_when_angle_just_inside_deadzone(self, node):
        """angle_rad = 0.051 < 0.052 deadzone → raw_ang = 0."""
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.051)
        assert node.last_ang == 0.0

    def test_no_angular_output_negative_inside_deadzone(self, node):
        node.prev_ang = 0.0
        node.control_robot(2.0, -0.051)
        assert node.last_ang == 0.0

    def test_angular_output_when_angle_exactly_at_deadzone_boundary(self, node):
        """angle_rad = 0.052 is NOT < deadzone, so angular command is produced."""
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.052)
        assert node.last_ang != 0.0

    def test_angular_output_when_angle_above_deadzone(self, node):
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.5)
        assert node.last_ang != 0.0

    def test_negative_angle_produces_negative_angular_output(self, node):
        node.prev_ang = 0.0
        node.control_robot(2.0, -0.5)
        assert node.last_ang < 0.0

    def test_positive_angle_produces_positive_angular_output(self, node):
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.5)
        assert node.last_ang > 0.0


class TestControlRobotVelocitySmoothing:
    def test_output_is_blend_of_prev_and_raw(self, node):
        """First call with prev=0: smooth_lin = alpha * raw + (1-alpha) * 0 = alpha * raw."""
        node.prev_lin = 0.0
        node.prev_ang = 0.0
        alpha = node.smooth_alpha  # 0.35
        distance = 2.0  # error_d = 1.0, raw_lin = k_lin * 1.0 = 0.7, clamped to 0.2
        node.control_robot(distance, 0.0)
        expected_lin = alpha * 0.2  # clamped max_lin_speed
        assert abs(node.last_lin - expected_lin) < 1e-9

    def test_second_call_uses_previous_smoothed_value(self, node):
        """Second call should blend using the prev_lin saved from first call."""
        node.prev_lin = 0.0
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.0)
        first_lin = node.last_lin
        # Second call: same inputs, prev_lin is now first_lin
        node.control_robot(2.0, 0.0)
        alpha = node.smooth_alpha
        raw = 0.2  # max_lin_speed
        expected = alpha * raw + (1.0 - alpha) * first_lin
        assert abs(node.last_lin - expected) < 1e-9

    def test_prev_values_updated_after_call(self, node):
        node.prev_lin = 0.0
        node.prev_ang = 0.0
        node.control_robot(2.0, 0.5)
        assert node.prev_lin == node.last_lin
        assert node.prev_ang == node.last_ang

    def test_output_not_instantaneous_jump(self, node):
        """With prev=0 and large command, output < raw (smoothing attenuates)."""
        node.prev_lin = 0.0
        node.control_robot(2.0, 0.0)
        raw_lin = 0.2  # clamped
        assert node.last_lin < raw_lin  # alpha=0.35 < 1.0


class TestControlRobotAlignedState:
    def test_aligned_status_at_target_distance_zero_angle(self, node):
        """At target_distance with no angle error → both velocities ≈0 → ALIGNED."""
        node.prev_lin = 0.0
        node.prev_ang = 0.0
        node.control_robot(node.target_distance, 0.0)
        assert node.last_status == "ALIGNED - HOLDING"

    def test_following_status_when_far_away(self, node):
        node.control_robot(3.0, 0.0)
        assert node.last_status == "FOLLOWING TARGET"


class TestControlRobotMaxSpeedClamping:
    def test_linear_never_exceeds_max(self, node):
        node.prev_lin = 0.0
        # Very far away — raw_lin would exceed max_lin_speed without clamping
        node.control_robot(100.0, 0.0)
        # smooth_lin = alpha * max_lin_speed (since prev=0)
        assert node.last_lin <= node.max_lin_speed

    def test_angular_never_exceeds_max(self, node):
        node.prev_ang = 0.0
        # Large angle → raw_ang clamped to max_ang_speed
        node.control_robot(2.0, math.pi)
        assert abs(node.last_ang) <= node.max_ang_speed


# ===========================================================================
# search_for_target()
# ===========================================================================


class TestSearchForTargetActuationDisabled:
    def test_publishes_zero_twist_when_disabled(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.linear.x == 0.0
        assert twist.angular.z == 0.0

    def test_sets_zero_last_values_when_disabled(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        assert node.last_lin == 0.0
        assert node.last_ang == 0.0

    def test_sets_disabled_status(self, node):
        node.actuation_enabled = False
        node.search_for_target()
        assert node.last_status == "SEARCHING (disabled)"


class TestSearchForTargetLinearAlwaysZero:
    def test_linear_x_is_zero_during_normal_search(self, node):
        node.actuation_enabled = True
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_lin == 0.0

    def test_linear_x_is_zero_when_search_complete(self, node):
        node.actuation_enabled = True
        node.search_angle_rad = 2.0 * math.pi  # already done
        node.search_for_target()
        assert node.last_lin == 0.0


class TestSearchForTargetTurnDirection:
    def test_positive_last_angle_error_produces_positive_angular(self, node):
        """Positive last_angle_error → direction = +1 → positive angular command."""
        node.last_angle_error = 0.3
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang > 0.0

    def test_negative_last_angle_error_produces_negative_angular(self, node):
        """Negative last_angle_error → direction = -1 → negative angular command."""
        node.last_angle_error = -0.3
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang < 0.0

    def test_zero_last_angle_error_treated_as_positive(self, node):
        """last_angle_error == 0 → direction = +1 (>=0 condition)."""
        node.last_angle_error = 0.0
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert node.last_ang >= 0.0


class TestSearchForTargetFullTurnStop:
    def test_stops_after_360_degrees_integrated(self, node):
        """search_angle_rad >= 2*pi → publishes zero angular and sets DONE status."""
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        assert node.last_ang == 0.0

    def test_status_after_360_degrees(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        assert "360" in node.last_status or "DONE" in node.last_status

    def test_publishes_zero_twist_after_360_degrees(self, node):
        node.search_angle_rad = 2.0 * math.pi
        node.search_for_target()
        node.cmd_pub.publish.assert_called_once()
        twist = node.cmd_pub.publish.call_args[0][0]
        assert twist.angular.z == 0.0

    def test_angular_nonzero_just_before_360_threshold(self, node):
        node.search_angle_rad = 2.0 * math.pi - 0.01
        node.search_for_target()
        # Should still be turning (not stopped)
        assert node.last_ang != 0.0

    def test_search_angle_accumulates_over_time(self, node):
        """Each call advances search_angle_rad by abs(w)*dt."""
        node.search_angle_rad = 0.0
        node._search_prev_t = None
        node.search_for_target()  # initialises _search_prev_t
        initial_angle = node.search_angle_rad
        # Simulate a dt by backdating _search_prev_t
        node._search_prev_t = time.time() - 1.0
        node.search_for_target()
        assert node.search_angle_rad > initial_angle


class TestSearchForTargetSearchingStatus:
    def test_status_is_searching_during_normal_rotation(self, node):
        node.search_angle_rad = 0.0
        node.search_for_target()
        assert "SEARCHING" in node.last_status


# ===========================================================================
# _shirt_color_ratio(mask_color_frame, body, img_h, img_w)
# ===========================================================================

IMG_H = 720
IMG_W = 1280


def _make_body(x1=100, y1=100, x2=300, y2=500, mask=None):
    """Helper: create a mock body with a rectangular bounding_box_2d."""
    body = MagicMock()
    body.bounding_box_2d = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]
    body.mask = mask
    return body


class TestShirtColorRatioZeroArea:
    def test_returns_zero_when_bbox_has_zero_width(self, node):
        body = _make_body(x1=200, y1=100, x2=200, y2=400)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0

    def test_returns_zero_when_bbox_has_zero_height(self, node):
        # With only 55% of height, y2 = y1 + 0.55*(y1-y1) = y1 → zero height crop
        body = _make_body(x1=100, y1=200, x2=300, y2=200)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0

    def test_returns_zero_on_invalid_body_exception(self, node):
        """body.bounding_box_2d raises → outer except → returns 0.0."""
        body = MagicMock()
        body.bounding_box_2d = MagicMock(side_effect=RuntimeError("bad body"))
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0


class TestShirtColorRatioAllWhiteMask:
    def test_all_white_mask_no_segmentation_returns_one(self, node):
        """All-white mask in bbox with no segmentation mask → ratio = 1.0."""
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        body = _make_body(100, 100, 300, 500)
        body.mask = None  # no segmentation
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert abs(result - 1.0) < 1e-6

    def test_all_black_mask_no_segmentation_returns_zero(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        body = _make_body(100, 100, 300, 500)
        body.mask = None
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0


class TestShirtColorRatioPartialMask:
    def test_half_white_mask_returns_approx_half(self, node):
        """Left half of image is white, right half is black.
        If bbox is entirely within the white region, ratio ~ 1.0.
        If bbox straddles midpoint equally, ratio ~ 0.5.
        """
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        mid = IMG_W // 2
        mask[:, :mid] = 255  # left half white

        # Place bbox centered on midpoint so approximately half is white
        # x1..x2 = mid-100..mid+100
        body = _make_body(mid - 100, 100, mid + 100, 500)
        body.mask = None

        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert 0.45 <= result <= 0.55

    def test_fully_in_white_region_returns_one(self, node):
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        body = _make_body(100, 100, 400, 600)
        body.mask = None
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert abs(result - 1.0) < 1e-6

    def test_fully_in_black_region_returns_zero(self, node):
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        body = _make_body(100, 100, 400, 600)
        body.mask = None
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0


class TestShirtColorRatioUpperCrop:
    def test_only_upper_55_percent_sampled(self, node):
        """Color mask is white only in the bottom 50% of bbox; upper 55% crop
        should miss it → ratio = 0."""
        body = _make_body(x1=100, y1=100, x2=300, y2=500)  # height = 400 px
        # y2 (upper-55% boundary) = 100 + int(0.55 * 400) = 100 + 220 = 320
        # Paint white only below row 320 in the bbox
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        mask[320:, 100:300] = 255  # rows 320..500, entirely below crop boundary

        body.mask = None
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 0.0

    def test_color_in_upper_55_percent_is_detected(self, node):
        """White mask only in the upper 55% crop region → ratio = 1.0."""
        body = _make_body(x1=100, y1=100, x2=300, y2=500)  # height = 400 px
        # Crop covers rows 100..319 (y2 = 100 + 220 = 320)
        mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        mask[100:320, 100:300] = 255  # exactly the crop region

        body.mask = None
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert abs(result - 1.0) < 1e-6


class TestShirtColorRatioSegmentationMask:
    def test_segmentation_mask_gates_pixel_count(self, node):
        """When segmentation mask covers only half the bbox pixels, denominator
        is that half; color-AND-mask nonzero count governs the ratio."""
        body = _make_body(x1=0, y1=0, x2=100, y2=200)
        # Upper 55%: y2_crop = 0 + int(0.55*200) = 110, so roi is rows 0:110, cols 0:100
        roi_h = 110
        roi_w = 100

        color_mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)  # all white

        # Segmentation mask: only left half of image has person pixels
        seg_data = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        seg_data[:, : IMG_W // 2] = 200  # left half

        seg_mock = MagicMock()
        seg_mock.get_data.return_value = seg_data
        body.mask = seg_mock

        # bitwise_and(color_roi, seg_roi): both nonzero in left half of roi
        # seg_roi is seg_data[0:110, 0:100] = all 200 (within left half)
        # → person_mask roi = all 255; denom = roi_h * roi_w = 11000
        # bitwise_and = color_roi & seg_roi = all-255 & all-255 = all-255
        # ratio = 11000 / 11000 = 1.0

        # Patch cv2.bitwise_and to return an all-nonzero array of matching shape
        expected_roi_shape = (roi_h, roi_w)
        cv2_mock.bitwise_and.side_effect = lambda a, b: np.ones(a.shape, dtype=np.uint8) * 255

        result = node._shirt_color_ratio(color_mask, body, IMG_H, IMG_W)
        assert result > 0.0  # segmentation path executed and returned a valid ratio

    def test_segmentation_mask_exception_falls_back_to_raw_bbox(self, node):
        """If segmentation mask access raises, falls back to raw bbox ratio."""
        body = _make_body(x1=100, y1=100, x2=300, y2=500)
        mask_full_white = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)

        seg_mock = MagicMock()
        seg_mock.get_data.side_effect = RuntimeError("mock seg error")
        body.mask = seg_mock

        # get_logger().debug() must not crash
        node.get_logger = MagicMock(return_value=MagicMock())

        result = node._shirt_color_ratio(mask_full_white, body, IMG_H, IMG_W)
        assert abs(result - 1.0) < 1e-6

    def test_no_segmentation_mask_uses_raw_bbox(self, node):
        """body.mask = None → skip segmentation → raw bbox ratio computed."""
        body = _make_body(x1=100, y1=100, x2=300, y2=500)
        body.mask = None
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert result == 1.0


class TestShirtColorRatioEdgeCases:
    def test_bbox_clipped_to_image_boundary(self, node):
        """Bbox that extends beyond image dimensions should be clipped, not crash."""
        body = _make_body(x1=-50, y1=-50, x2=IMG_W + 100, y2=IMG_H + 100)
        body.mask = None
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert 0.0 <= result <= 1.0

    def test_single_pixel_bbox(self, node):
        """1x1 bbox → y2 = y1 + 0 → zero height after 55% crop → 0.0."""
        body = _make_body(x1=100, y1=100, x2=101, y2=101)
        body.mask = None
        mask = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        # y2_full = 101, y2 = 100 + int(0.55 * 1) = 100 → y2 == y1 → 0.0
        assert result == 0.0

    def test_result_always_in_range_zero_to_one(self, node):
        """Invariant: result in [0, 1] regardless of inputs."""
        body = _make_body(50, 50, 400, 600)
        body.mask = None
        mask = np.random.randint(0, 256, (IMG_H, IMG_W), dtype=np.uint8)
        result = node._shirt_color_ratio(mask, body, IMG_H, IMG_W)
        assert 0.0 <= result <= 1.0
