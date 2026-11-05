import config as cfg
from src.density_estimator import classify


def green_time(level):
    return min(max(cfg.GREEN_TIME[level], cfg.MIN_GREEN), cfg.MAX_GREEN)


def choose_phase(loads, waited, last=None):
    starved = [a for a in waited if waited[a] > cfg.MAX_RED]
    if starved:
        return max(starved, key=waited.get), "starvation limit reached"
    candidates = [a for a in loads if a != last] or list(loads)
    return max(candidates, key=lambda a: (loads[a], waited[a])), "highest density"


class AdaptiveController:
    def __init__(self, approaches):
        self.approaches = list(approaches)
        self.last = None
        self.last_green_end = dict.fromkeys(self.approaches, 0)

    def next_phase(self, loads, t):
        waited = {a: t - self.last_green_end[a] for a in self.approaches}
        approach, _ = choose_phase(loads, waited, self.last)
        green = green_time(classify(loads[approach]))
        self.last = approach
        self.last_green_end[approach] = t + cfg.LOST_TIME + green
        return approach, green


class FixedController:
    def __init__(self, approaches, green=cfg.DEFAULT_GREEN):
        self.approaches, self.green, self.index = list(approaches), green, -1

    def next_phase(self, loads, t):
        self.index = (self.index + 1) % len(self.approaches)
        return self.approaches[self.index], self.green
