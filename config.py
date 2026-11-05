from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "yolo11n.pt"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

# COCO class ids. Add e.g. 3: "motorcycle" here and in VEHICLE_WEIGHTS to support more classes.
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
VEHICLE_WEIGHTS = {"car": 1.0, "bus": 3.0, "truck": 2.5}  # approximate PCU (passenger car units)

DETECT_CONF = 0.1  # low on purpose: ByteTrack also uses low-score boxes to keep tracks alive
IMG_SIZE = 640
FRAME_SKIP = 3
RESIZE_WIDTH = 960

TRACK_HIGH_THRESH = 0.25
TRACK_LOW_THRESH = 0.1
NEW_TRACK_THRESH = 0.25
TRACK_LOST_SECONDS = 2  # a vanished vehicle can be re-identified for this long
MATCH_THRESH = 0.8

LOST_PATIENCE = 10  # processed frames a track may vanish before the vehicle is treated as gone
MIN_FRAMES_IN_ROI = 3  # ignore tracks seen fewer times than this (noise)
MIN_MOVE_PX = 30  # smaller displacement inside the ROI = "stationary"

SMOOTH_WINDOW = 10  # processed frames averaged for density
DENSITY_LIMITS = {"Low": 5, "Medium": 12, "High": 20}  # upper PCU limit per level, above = Critical
LEVELS = [*DENSITY_LIMITS, "Critical"]
PRIORITY = {"Low": 4, "Medium": 3, "High": 2, "Critical": 1}  # 1 = served first

APPROACHES = ["North", "South", "East", "West"]
GREEN_TIME = {"Low": 10, "Medium": 25, "High": 40, "Critical": 60}  # seconds
MIN_GREEN = 10
MAX_GREEN = 60
DEFAULT_GREEN = 30  # fixed-time green
MAX_RED = 120  # an approach waiting longer than this is served next
LOST_TIME = 4  # yellow + all-red + start-up per phase

SATURATION_FLOW = 0.5  # vehicles per second discharged during green
SIM_DURATION = 3600
SIM_SEEDS = 5
DEFAULT_RATES = {"North": 8.0, "South": 5.0, "East": 3.0, "West": 2.0}  # vehicles per minute
