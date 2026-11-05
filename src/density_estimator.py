from collections import deque
import config as cfg


def weighted_load(counts, weights=cfg.VEHICLE_WEIGHTS):
    return sum(weights[name] * n for name, n in counts.items())


def classify(load, limits=cfg.DENSITY_LIMITS):
    for level, upper in limits.items():
        if load <= upper:
            return level
    return "Critical"


class DensityEstimator:
    def __init__(self, window=cfg.SMOOTH_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, present_counts):
        self.history.append(weighted_load(present_counts))
        load = sum(self.history) / len(self.history)
        return load, classify(load)
