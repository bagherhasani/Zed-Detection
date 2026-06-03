import cv2
import numpy as np
import pyzed.sl as sl


def compute_color_mask(frame_bgr, lower, upper):
    # convert to HSV because light changes
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # white pixcels matches, black everywhere else
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def get_torso_polygon(keypoints_2d, img_w, img_h):
    # get pixel coordinates of the four torso corners from the ZED skeleton
    LS = sl.BODY_38_PARTS.LEFT_SHOULDER.value
    RS = sl.BODY_38_PARTS.RIGHT_SHOULDER.value
    LH = sl.BODY_38_PARTS.LEFT_HIP.value
    RH = sl.BODY_38_PARTS.RIGHT_HIP.value

    pts = []
    for idx in (LS, RS, RH, LH):
        x, y = keypoints_2d[idx]
        if x <= 0 or y <= 0 or x >= img_w or y >= img_h:
            return None  # joint is missing or out of frame
        pts.append([int(x), int(y)])

    return np.array(pts, dtype=np.int32)


def color_ratio(mask_color, poly_pts):
    if poly_pts is None or mask_color is None:
        return 0.0

    # draw the torso shape 
    poly_mask = np.zeros(mask_color.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_pts], 255)

    area = int(np.count_nonzero(poly_mask))
    if area == 0:
        return 0.0

    # pixels that are inside the torso AND match the colour
    colored = cv2.bitwise_and(mask_color, poly_mask)
    ratio = float(np.count_nonzero(colored)) / float(area)
    return ratio
