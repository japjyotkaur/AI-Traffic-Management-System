from types import SimpleNamespace
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
import config as cfg


class Detections:
    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


class Tracker:
    def __init__(self, fps=10):
        args = SimpleNamespace(
            track_high_thresh=cfg.TRACK_HIGH_THRESH,
            track_low_thresh=cfg.TRACK_LOW_THRESH,
            new_track_thresh=cfg.NEW_TRACK_THRESH,
            track_buffer=max(1, int(cfg.TRACK_LOST_SECONDS * fps)),
            match_thresh=cfg.MATCH_THRESH,
            fuse_score=True,
        )
        self.bytetrack = BYTETracker(args)

    def update(self, detections, frame):
        x1, y1, x2, y2 = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
        xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
        return self.bytetrack.update(Detections(xywh, detections[:, 4], detections[:, 5]), frame)  # rows: x1, y1, x2, y2, id, score, class, index
