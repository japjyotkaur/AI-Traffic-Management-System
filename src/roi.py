import math
import cv2
import numpy as np
import config as cfg


class ROI:
    def __init__(self, points=None, width=1, height=1):
        self.polygon = None if points is None else (np.array(points) * [width, height]).astype(np.int32)

    def contains(self, point):
        if self.polygon is None:
            return True
        return cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False) >= 0

    def draw(self, frame):
        if self.polygon is not None:
            cv2.polylines(frame, [self.polygon], True, (255, 200, 0), 2)


def parse_points(text):
    if not text.strip():
        return None
    try:
        points = [tuple(map(float, pair.split(","))) for pair in text.split(";") if pair.strip()]
    except ValueError:
        raise ValueError("ROI format: x1,y1; x2,y2; x3,y3; ...")
    if len(points) < 3 or any(len(p) != 2 or not all(0 <= v <= 1 for v in p) for p in points):
        raise ValueError("ROI needs at least 3 points, each with x and y between 0 and 1")
    return points


def movement(start, end, min_move=cfg.MIN_MOVE_PX):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < min_move:
        return "stationary"
    if abs(dy) >= abs(dx):
        return "toward_camera" if dy > 0 else "away_from_camera"
    return "right" if dx > 0 else "left"
