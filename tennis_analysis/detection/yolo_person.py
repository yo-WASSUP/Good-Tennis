import os
from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec

from .device import is_coreml_model, select_yolo_device


class YOLOPersonDetector:
    """Ultralytics YOLO person detector with optional built-in MOT tracking."""

    def __init__(self, model_path="weights/yolo26s.pt", device="auto", conf=0.25,
                 tracker="none"):
        self.model_path = model_path
        self.conf = conf
        self.tracker = tracker
        self.tracker_config = self._tracker_config_from_name(tracker)
        self._require_tracker_dependencies()
        self.inference_name = "YOLO-Person" if self.tracker == "none" else f"YOLO-Person-{self.tracker}"

        from ultralytics import YOLO

        self.device, self.half = select_yolo_device(device, model_path=self.model_path)

        tracker_label = self.tracker_config if self.tracker != "none" else "none"
        print(f"Initializing YOLO person model (model: {self.model_path}, device: {self.device}, tracker: {tracker_label})")
        self.model = YOLO(self.model_path)
        self._fuse_model()
        self._warmup_model()

    def process_frame(self, frame):
        result = self._run_inference(frame)
        if result.boxes is None or result.boxes.xyxy is None or result.boxes.xyxy.shape[0] == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
        track_ids = None
        if getattr(result.boxes, "id", None) is not None:
            track_ids = result.boxes.id.detach().cpu().numpy()

        boxes = []
        for idx, box in enumerate(xyxy):
            score = float(conf[idx]) if conf is not None else None
            track_id = int(track_ids[idx]) if track_ids is not None else None
            boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), score, track_id))
        return boxes

    def _run_inference(self, frame):
        if self.tracker == "none":
            return self.model(frame, conf=self.conf, device=self.device, classes=[0], half=self.half, verbose=False)[0]

        try:
            return self.model.track(
                frame,
                conf=self.conf,
                device=self.device,
                classes=[0],
                half=self.half,
                verbose=False,
                persist=True,
                tracker=self.tracker_config,
            )[0]
        except Exception as exc:
            raise RuntimeError(
                f"YOLO person tracker failed: {exc}. "
                "Fix the tracker dependency or run with --person-tracker none."
            ) from exc

    def _tracker_config_from_name(self, tracker):
        if tracker == "botsort":
            return "botsort.yaml"
        if tracker == "bytetrack":
            return "bytetrack.yaml"
        return ""

    def _require_tracker_dependencies(self):
        if self.tracker == "none":
            return
        if find_spec("lap") is None:
            raise RuntimeError(
                "Missing required dependency for YOLO person tracking: lap>=0.5.12.\n"
                "Install it with one of these commands:\n"
                "  pip install -r requirements.txt\n"
                "  pip install \"lap>=0.5.12\"\n"
                "Or disable MOT tracking with: --person-tracker none"
            )

    def _fuse_model(self):
        try:
            import logging
            from ultralytics.utils import LOGGER

            previous_level = LOGGER.level
            LOGGER.setLevel(logging.ERROR)
            try:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    self.model.fuse()
            finally:
                LOGGER.setLevel(previous_level)
        except Exception:
            pass

    def _warmup_model(self):
        if self.device == "cpu" and not is_coreml_model(self.model_path):
            return
        try:
            import logging
            import numpy as np
            from ultralytics.utils import LOGGER

            previous_level = LOGGER.level
            LOGGER.setLevel(logging.ERROR)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    self._run_inference(dummy)
                    # Initialize tracking callbacks outside the frame loop, then
                    # discard the dummy frame's tracks and camera-motion state.
                    if self.tracker != "none":
                        for tracker in self.model.predictor.trackers:
                            tracker.reset()
            finally:
                LOGGER.setLevel(previous_level)
        except Exception:
            pass
