import numpy as np
from ultralytics import YOLO
import config as cfg


class Detector:
    def __init__(self, model_path=cfg.MODEL_PATH, classes=cfg.VEHICLE_CLASSES, conf=cfg.DETECT_CONF, imgsz=cfg.IMG_SIZE):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model_path))  # downloads yolo11n.pt on first run if missing
        self.class_ids = list(classes)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        result = self.model.predict(frame, classes=self.class_ids, conf=self.conf, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        boxes = result.boxes
        return np.hstack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]])
