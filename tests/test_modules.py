import unittest
import numpy as np
import config as cfg
from src.density_estimator import DensityEstimator, classify, weighted_load
from src.performance import compare
from src.roi import ROI, movement, parse_points
from src.signal_controller import AdaptiveController, FixedController, choose_phase, green_time
from src.simulation import generate_arrivals, simulate
from src.vehicle_counter import VehicleCounter


def track(track_id, x, y_bottom, class_id=2):
    return [x - 20, y_bottom - 40, x + 20, y_bottom, track_id, 0.9, class_id, 0]


class TestDetector(unittest.TestCase):
    @unittest.skipUnless(cfg.MODEL_PATH.exists(), "run the app once to download models/yolo11n.pt")
    def test_detects_bus_in_sample_image(self):
        import cv2
        from ultralytics.utils import ASSETS
        from src.detector import Detector
        detections = Detector().detect(cv2.imread(str(ASSETS / "bus.jpg")))
        self.assertEqual(detections.shape[1], 6)
        self.assertIn(5, detections[:, 5])
        self.assertTrue(set(detections[:, 5]) <= set(cfg.VEHICLE_CLASSES))


class TestTracker(unittest.TestCase):
    def setUp(self):
        from src.tracker import Tracker
        self.tracker = Tracker(fps=10)
        self.frame = np.zeros((540, 960, 3), dtype=np.uint8)

    def run_frames(self, detections_per_frame):
        return [self.tracker.update(np.array(d, dtype=float), self.frame) for d in detections_per_frame]

    def test_same_vehicle_keeps_same_id(self):
        frames = [[[100 + 8 * i, 200, 180 + 8 * i, 260, 0.9, 2]] for i in range(20)]
        ids = {int(t[4]) for out in self.run_frames(frames) for t in out}
        self.assertEqual(len(ids), 1)

    def test_two_vehicles_get_two_ids(self):
        frames = [[[100 + 8 * i, 100, 180 + 8 * i, 160, 0.9, 2], [600 - 8 * i, 300, 700 - 8 * i, 380, 0.8, 7]] for i in range(15)]
        ids = {int(t[4]) for out in self.run_frames(frames) for t in out}
        self.assertEqual(len(ids), 2)

    def test_empty_detections(self):
        self.assertEqual(len(self.run_frames([np.zeros((0, 6))])[0]), 0)


class TestROIAndCounter(unittest.TestCase):
    def setUp(self):
        self.roi = ROI([(0, 0.3), (1, 0.3), (1, 0.7), (0, 0.7)], 1000, 1000)

    def test_roi_contains(self):
        self.assertTrue(self.roi.contains((500, 500)))
        self.assertFalse(self.roi.contains((500, 100)))
        self.assertTrue(ROI().contains((5, 5)))

    def test_parse_points(self):
        self.assertEqual(parse_points("0,0; 1,0; 1,1"), [(0, 0), (1, 0), (1, 1)])
        self.assertIsNone(parse_points(""))
        for bad in ("0,0; 1,1", "a,b; 1,0; 1,1", "0,0; 2,0; 1,1"):
            with self.assertRaises(ValueError):
                parse_points(bad)

    def test_movement(self):
        self.assertEqual(movement((0, 0), (0, 100)), "toward_camera")
        self.assertEqual(movement((0, 100), (0, 0)), "away_from_camera")
        self.assertEqual(movement((0, 0), (100, 10)), "right")
        self.assertEqual(movement((0, 0), (5, 5)), "stationary")

    def test_vehicle_counted_once_after_leaving_roi(self):
        counter = VehicleCounter(self.roi)
        for frame, y in enumerate(range(100, 950, 50)):
            counter.update([track(1, 500, y)], frame)
        counter.update([track(1, 500, 300)], 40)  # same id comes back: must not be counted again
        self.assertEqual(counter.passed["car"], 1)
        self.assertEqual(counter.log[0]["movement"], "toward_camera")

    def test_present_and_passed_are_separate(self):
        counter = VehicleCounter(self.roi)
        counter.update([track(1, 300, 500), track(2, 600, 520, 7)], 0)
        self.assertEqual(counter.present, {"car": 1, "bus": 0, "truck": 1})
        self.assertEqual(sum(counter.passed.values()), 0)

    def test_short_noise_track_is_ignored(self):
        counter = VehicleCounter(self.roi)
        counter.update([track(1, 500, 500)], 0)
        counter.update([track(1, 500, 900)], 1)
        self.assertEqual(sum(counter.passed.values()), 0)

    def test_without_roi_vehicle_counted_after_it_disappears(self):
        counter = VehicleCounter(ROI())
        for frame in range(5):
            counter.update([track(7, 100 + 10 * frame, 300, 5)], frame)
        for frame in range(5, 5 + cfg.LOST_PATIENCE + 2):
            counter.update([], frame)
        self.assertEqual(counter.passed["bus"], 1)

    def test_majority_class_vote(self):
        counter = VehicleCounter(self.roi)
        for frame, class_id in enumerate([2, 7, 7, 7, 2]):
            counter.update([track(1, 500, 400 + 20 * frame, class_id)], frame)
        counter.update([], 100)
        self.assertEqual(counter.passed["truck"], 1)


class TestDensity(unittest.TestCase):
    def test_weighted_load(self):
        self.assertEqual(weighted_load({"car": 2, "bus": 1, "truck": 0}), 5)

    def test_levels(self):
        self.assertEqual([classify(x) for x in (3, 8, 15, 25)], ["Low", "Medium", "High", "Critical"])

    def test_smoothing(self):
        estimator = DensityEstimator(window=2)
        estimator.update({"car": 0})
        load, _ = estimator.update({"car": 10})
        self.assertEqual(load, 5)


class TestSignal(unittest.TestCase):
    def test_green_time_increases_with_density_and_stays_in_limits(self):
        times = [green_time(level) for level in cfg.LEVELS]
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(cfg.MIN_GREEN <= t <= cfg.MAX_GREEN for t in times))

    def test_busiest_approach_chosen(self):
        loads = {"North": 3, "South": 15, "East": 1, "West": 0}
        self.assertEqual(choose_phase(loads, dict.fromkeys(loads, 10))[0], "South")

    def test_starved_approach_overrides_density(self):
        loads = {"North": 30, "South": 1, "East": 1, "West": 1}
        waited = {"North": 5, "South": cfg.MAX_RED + 1, "East": 10, "West": 10}
        self.assertEqual(choose_phase(loads, waited)[0], "South")

    def test_no_approach_ignored_indefinitely(self):
        controller, t, last_end = AdaptiveController(cfg.APPROACHES), 0, dict.fromkeys(cfg.APPROACHES, 0)
        loads = {"North": 40, "South": 1, "East": 1, "West": 1}
        for _ in range(60):
            approach, green = controller.next_phase(loads, t)
            self.assertLessEqual(t - last_end[approach], cfg.MAX_RED + cfg.MAX_GREEN + 2 * cfg.LOST_TIME)
            t += cfg.LOST_TIME + green
            last_end[approach] = t

    def test_fixed_controller_cycles_in_order(self):
        controller = FixedController(cfg.APPROACHES, 30)
        self.assertEqual([controller.next_phase({}, 0) for _ in range(5)], [(a, 30) for a in cfg.APPROACHES] + [("North", 30)])


class TestSimulation(unittest.TestCase):
    def test_vehicle_conservation_and_valid_waits(self):
        arrivals = generate_arrivals(cfg.DEFAULT_RATES, 1200, seed=1)
        for controller in (FixedController(cfg.APPROACHES), AdaptiveController(cfg.APPROACHES)):
            result = simulate(controller, arrivals, 1200)
            self.assertEqual(len(result["waits"]) + len(result["remaining"]), result["arrived"])
            self.assertTrue(all(w > 0 for w in result["waits"] + result["remaining"]))

    def test_compare_table(self):
        table, queues = compare(cfg.DEFAULT_RATES, duration=600, seeds=2)
        self.assertEqual(list(table.columns), ["Fixed-time", "Adaptive", "Change (%)", "Better"])
        self.assertEqual(set(queues), {"Fixed-time", "Adaptive"})

    def test_zero_demand_does_not_crash(self):
        table, _ = compare(dict.fromkeys(cfg.APPROACHES, 0.0), duration=300, seeds=1)
        self.assertEqual(table.loc["Throughput (veh/hour)", "Adaptive"], 0)


if __name__ == "__main__":
    unittest.main()
