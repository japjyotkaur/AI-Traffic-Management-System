import argparse
import sys
from pathlib import Path
import cv2
import pandas as pd
import config as cfg
from src.density_estimator import DensityEstimator
from src.detector import Detector
from src.roi import ROI, parse_points
from src.signal_controller import green_time
from src.tracker import Tracker
from src.vehicle_counter import VehicleCounter
from src.video_reader import read_frames, video_info

COLORS = {"car": (0, 200, 0), "bus": (0, 140, 255), "truck": (0, 0, 220)}


class Pipeline:
    def __init__(self, video_path, roi_points=None, skip=cfg.FRAME_SKIP, detector=None):
        info = video_info(video_path)
        height = int(info["height"] * cfg.RESIZE_WIDTH / info["width"])
        self.video_path, self.skip, self.fps = video_path, skip, info["fps"]
        self.detector = detector or Detector()
        self.tracker = Tracker(self.fps / skip)
        self.roi = ROI(roi_points, cfg.RESIZE_WIDTH, height)
        self.counter = VehicleCounter(self.roi)
        self.estimator = DensityEstimator()

    def run(self, max_frames=None):
        for index, frame in read_frames(self.video_path, self.skip, cfg.RESIZE_WIDTH, max_frames):
            raw = frame.copy()
            tracks = self.tracker.update(self.detector.detect(frame), frame)
            present = self.counter.update(tracks, index)
            load, level = self.estimator.update(present)
            stats = {
                "frame": index,
                "time_s": index / self.fps,
                "present": dict(present),
                "passed": dict(self.counter.passed),
                "load": load,
                "level": level,
                "priority": cfg.PRIORITY[level],
                "green": green_time(level),
            }
            yield raw, self.draw(frame, tracks, stats), stats

    def draw(self, frame, tracks, stats):
        self.roi.draw(frame)
        for x1, y1, x2, y2, track_id, score, class_id, _ in tracks:
            name = cfg.VEHICLE_CLASSES[int(class_id)]
            color = COLORS.get(name, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{name} #{int(track_id)} {score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        lines = [
            f"In ROI: {sum(stats['present'].values())}   Passed: {sum(stats['passed'].values())}",
            f"Density: {stats['level']} ({stats['load']:.1f} PCU)",
            f"Suggested green: {stats['green']} s",
        ]
        cv2.rectangle(frame, (0, 0), (320, 70), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame


def save_results(pipeline, history):
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    stem = Path(pipeline.video_path).stem
    frames_csv, vehicles_csv = cfg.OUTPUT_DIR / f"{stem}_frames.csv", cfg.OUTPUT_DIR / f"{stem}_vehicles.csv"
    pd.json_normalize(history).to_csv(frames_csv, index=False)
    pd.DataFrame(pipeline.counter.log).to_csv(vehicles_csv, index=False)
    return frames_csv, vehicles_csv


def main():
    parser = argparse.ArgumentParser(description="Run detection, tracking and counting on one video without the dashboard")
    parser.add_argument("video")
    parser.add_argument("--skip", type=int, default=cfg.FRAME_SKIP)
    parser.add_argument("--max-frames", type=int, default=0, help="processed frames, 0 = whole video")
    parser.add_argument("--roi", default="", help='normalised polygon, e.g. "0,0.3; 1,0.3; 1,1; 0,1"')
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    try:
        pipeline = Pipeline(Path(args.video), parse_points(args.roi), args.skip)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))
    writer, history = None, []
    for _, frame, stats in pipeline.run(args.max_frames or None):
        history.append(stats)
        if args.save_video:
            if writer is None:
                out_path = cfg.OUTPUT_DIR / f"{Path(args.video).stem}_annotated.mp4"
                cfg.OUTPUT_DIR.mkdir(exist_ok=True)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), pipeline.fps / args.skip, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    if writer:
        writer.release()
    if not history:
        sys.exit("No frames were processed")
    print("Frames processed:", len(history))
    print("Vehicles passed through ROI:", history[-1]["passed"])
    print("Saved:", *save_results(pipeline, history))


if __name__ == "__main__":
    main()
