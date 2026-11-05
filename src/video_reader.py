import cv2
import config as cfg


def list_videos(folder=cfg.VIDEO_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in cfg.VIDEO_EXTENSIONS)


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def video_info(path):
    cap = open_video(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def resize(frame, width):
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height))


def first_frame(path, width=cfg.RESIZE_WIDTH):
    cap = open_video(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Video has no readable frames: {path}")
    return resize(frame, width)


def read_frames(path, skip=1, width=cfg.RESIZE_WIDTH, max_frames=None):
    skip = max(1, int(skip))
    cap = open_video(path)
    index = yielded = 0
    try:
        while not max_frames or yielded < max_frames:
            if not cap.grab():
                break
            if index % skip == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield index, resize(frame, width)
                    yielded += 1
            index += 1
    finally:
        cap.release()
