class YOLOPersonDetector:
    """Ultralytics YOLO person detector returning bottom-center boxes."""

    def __init__(self, model_path="weights/yolo26s.pt", device="auto", conf=0.25):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf = conf
        self.inference_name = "YOLO-Person"
        if device in (None, "auto"):
            selected = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    selected = 0
            except Exception:
                selected = "cpu"
            self.device = selected
        else:
            self.device = device

        print(f"Initializing YOLO person model (model: {self.model_path}, device: {self.device})")
        self.model = YOLO(self.model_path)

    def process_frame(self, frame):
        result = self.model(frame, conf=self.conf, device=self.device, classes=[0], verbose=False)[0]
        if result.boxes is None or result.boxes.xyxy is None or result.boxes.xyxy.shape[0] == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else None
        boxes = []
        for idx, box in enumerate(xyxy):
            score = float(conf[idx]) if conf is not None else None
            boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), score))
        return boxes
