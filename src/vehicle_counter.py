from collections import Counter
import config as cfg
from src.roi import movement


class VehicleCounter:
    def __init__(self, roi, classes=cfg.VEHICLE_CLASSES, patience=cfg.LOST_PATIENCE, min_frames=cfg.MIN_FRAMES_IN_ROI):
        self.roi, self.classes = roi, classes
        self.patience, self.min_frames = patience, min_frames
        self.names = list(classes.values())
        self.active = {}
        self.counted = set()
        self.log = []
        self.present = dict.fromkeys(self.names, 0)
        self.passed = dict.fromkeys(self.names, 0)

    def update(self, tracks, frame_idx):
        self.present = dict.fromkeys(self.names, 0)
        for x1, y1, x2, y2, track_id, _, class_id, _ in tracks:
            track_id = int(track_id)
            if track_id in self.counted:
                continue
            point = ((x1 + x2) / 2, y2)  # bottom-centre = where the vehicle touches the road
            state = self.active.get(track_id)
            if self.roi.contains(point):
                if state is None:
                    state = self.active[track_id] = {"votes": Counter(), "start": point, "entry": frame_idx, "frames": 0}
                state["votes"][self.classes[int(class_id)]] += 1
                state.update(last=point, last_seen=frame_idx)
                state["frames"] += 1
                self.present[self._class_of(state)] += 1
            elif state:
                self._finish(track_id)
        for track_id, state in list(self.active.items()):
            if frame_idx - state["last_seen"] > self.patience:
                self._finish(track_id)
        return self.present

    def _class_of(self, state):
        return state["votes"].most_common(1)[0][0]

    def _finish(self, track_id):
        state = self.active.pop(track_id)
        if state["frames"] < self.min_frames:
            return
        name = self._class_of(state)
        self.counted.add(track_id)
        self.passed[name] += 1
        self.log.append({
            "id": track_id,
            "class": name,
            "movement": movement(state["start"], state["last"]),
            "entry_frame": state["entry"],
            "exit_frame": state["last_seen"],
        })
