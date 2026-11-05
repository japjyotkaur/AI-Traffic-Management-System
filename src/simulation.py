from collections import deque
import numpy as np
import config as cfg


def generate_arrivals(rates, duration, seed):
    rng = np.random.default_rng(seed)
    return {a: np.sort(rng.uniform(0, duration, rng.poisson(rate * duration / 60))) for a, rate in rates.items()}


def simulate(controller, arrivals, duration=cfg.SIM_DURATION):
    queues = {a: deque() for a in arrivals}
    next_index = dict.fromkeys(arrivals, 0)
    waits, queue_series = [], []
    current, green_start, phase_end, credit = None, 0, 0, 0
    for t in range(duration):
        for a, times in arrivals.items():
            while next_index[a] < len(times) and times[next_index[a]] < t + 1:
                queues[a].append(times[next_index[a]])
                next_index[a] += 1
        if t >= phase_end:
            current, green = controller.next_phase({a: len(q) for a, q in queues.items()}, t)
            green_start = t + cfg.LOST_TIME
            phase_end = green_start + green
            credit = 0
        if t >= green_start:
            credit = min(credit + cfg.SATURATION_FLOW, 1)
            if credit >= 1 and queues[current]:
                waits.append(t + 1 - queues[current].popleft())
                credit -= 1
        queue_series.append(sum(len(q) for q in queues.values()))
    remaining = [duration - arrival for q in queues.values() for arrival in q]
    return {"waits": waits, "remaining": remaining, "queue": queue_series, "arrived": sum(len(v) for v in arrivals.values())}
