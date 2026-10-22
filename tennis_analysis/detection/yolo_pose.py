from .device import is_coreml_model, select_yolo_device


class YOLOPoseProcessor:
    """Ultralytics YOLO pose processor with COCO 17 keypoint output."""

    def __init__(self, model_path="weights/yolo11s-pose.pt", device="auto", conf=0.25):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf = conf
        self.inference_name = "YOLO-Pose"
        self.device, self.half = select_yolo_device(device, model_path=self.model_path)

        print(f"Initializing YOLO pose model (model: {self.model_path}, device: {self.device})")
        self.model = YOLO(self.model_path)
        self._fuse_model()
        self._warmup_model()

    def process_frame(self, frame):
        result = self.model(frame, conf=self.conf, device=self.device, half=self.half, verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            return None, None

        keypoints = result.keypoints.xy
        scores = result.keypoints.conf
        if keypoints.shape[0] == 0:
            return None, None

        keypoints = keypoints.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy() if scores is not None else None
        return keypoints, scores

    def _fuse_model(self):
        try:
            import os
            import logging
            from contextlib import redirect_stderr, redirect_stdout
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
            import os
            import numpy as np
            from contextlib import redirect_stderr, redirect_stdout
            from ultralytics.utils import LOGGER

            previous_level = LOGGER.level
            LOGGER.setLevel(logging.ERROR)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    self.model(dummy, conf=self.conf, device=self.device, half=self.half, verbose=False)
            finally:
                LOGGER.setLevel(previous_level)
        except Exception:
            pass

