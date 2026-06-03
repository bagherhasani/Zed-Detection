import cv2


def draw_person_overlay(img, body, poly):
    h, w = img.shape[:2]
    bb = body.bounding_box_2d

    # find the bounding box corners from the 4 points ZED gives us
    x1 = int(bb[0][0])
    y1 = int(bb[0][1])
    x2 = int(bb[2][0])
    y2 = int(bb[2][1])

    # clamp to image edges
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
    if poly is not None:
        cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, (0, 200, 255), 2)
    cv2.putText(img, "TARGET", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
