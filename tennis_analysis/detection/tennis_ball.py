from collections import deque
import time

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


class TennisBallTracker:
    """Detect, filter, track, and draw tennis ball positions."""

    def __init__(
        self,
        yolo_ball_model,
        trajectory_length=30,
        show_trajectory=True,
        show_performance_stats=False,
        max_jump_pixels=220,
        prediction_gate_pixels=260,
        max_missing_frames=5,
        roi_padding_ratio=0.08,
        max_box_area_ratio=0.004,
        max_aspect_ratio=4.0,
    ):
        self.yolo_ball_model = yolo_ball_model
        self.trajectory_length = trajectory_length
        self.show_trajectory = show_trajectory
        self.show_performance_stats = show_performance_stats
        self.max_jump_pixels = max_jump_pixels
        self.prediction_gate_pixels = prediction_gate_pixels
        self.max_missing_frames = max_missing_frames
        self.roi_padding_ratio = roi_padding_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.tennis_ball_trajectory = deque(maxlen=trajectory_length)
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0

        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            self.ultra_device = 0
        else:
            self.ultra_device = "cpu"

    def detect_ball(self, frame, conf=0.18, roi_corners=None):
        t0 = time.time()
        try:
            ball_results = self.yolo_ball_model(frame, conf=conf, device=self.ultra_device, verbose=False)[0]
        except TypeError:
            ball_results = self.yolo_ball_model(frame, conf=conf, verbose=False)[0]

        if self.show_performance_stats:
            print(f"YOLO tennis ball inference took {time.time() - t0:.2f} sec")

        candidates = self._extract_candidates(ball_results, frame.shape, roi_corners)
        selected = self._select_candidate(candidates)
        self.last_candidate = selected
        self.last_detection = {
            "visible": selected is not None,
            "accepted": False,
            "image": list(selected["point"]) if selected else None,
            "confidence": selected["confidence"] if selected else None,
            "candidate_count": len(candidates),
        }
        return list(selected["point"]) if selected else [0, 0]

    def update_trajectory(self, ball_position, roi_corners=None):
        if ball_position == [0, 0] or ball_position is None:
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        point = tuple(ball_position)
        if not self._point_in_roi(point, roi_corners):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        if self._is_outlier(point):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        self._append_valid_point(point)
        self.last_detection["accepted"] = True
        self.last_detection["image"] = list(point)
        return list(point)

    def _extract_candidates(self, ball_results, frame_shape, roi_corners):
        boxes = ball_results.boxes
        if boxes is None or boxes.xywh.shape[0] < 1:
            return []

        xywh = boxes.xywh.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
        frame_area = max(1, frame_shape[0] * frame_shape[1])

        candidates = []
        for box, confidence in zip(xywh, confidences):
            center_x, center_y, width, height = [float(value) for value in box]
            if width <= 0 or height <= 0:
                continue

            point = (int(center_x), int(center_y))
            area_ratio = (width * height) / frame_area
            aspect_ratio = max(width / height, height / width)
            if area_ratio > self.max_box_area_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if not self._point_in_roi(point, roi_corners):
                continue

            candidates.append(
                {
                    "point": point,
                    "confidence": float(confidence),
                    "area_ratio": float(area_ratio),
                    "aspect_ratio": float(aspect_ratio),
                }
            )

        return candidates

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        if not self.tennis_ball_trajectory:
            return max(candidates, key=lambda item: item["confidence"])

        predicted = self._predict_next_position()

        def score(candidate):
            distance = self._distance(candidate["point"], predicted)
            size_penalty = candidate["area_ratio"] * 4000
            return candidate["confidence"] * 1000 - distance * 1.4 - size_penalty

        return max(candidates, key=score)

    def _point_in_roi(self, point, roi_corners):
        if roi_corners is None:
            return True

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        padding = int(max(x2 - x1, y2 - y1) * self.roi_padding_ratio)
        return (x1 - padding) <= point[0] <= (x2 + padding) and (y1 - padding) <= point[1] <= (y2 + padding)

    def _is_outlier(self, point):
        if not self.tennis_ball_trajectory:
            return False

        last_point = self.tennis_ball_trajectory[-1]
        jump_distance = self._distance(point, last_point)
        strict_gate = self.missing_frames <= self.max_missing_frames
        if jump_distance > self.max_jump_pixels and strict_gate:
            return True

        predicted = self._predict_next_position()
        predicted_distance = self._distance(point, predicted)
        if predicted_distance > self.prediction_gate_pixels and strict_gate:
            return True

        return False

    def _predict_next_position(self):
        if len(self.tennis_ball_trajectory) < 2:
            return self.tennis_ball_trajectory[-1]

        prev_x, prev_y = self.tennis_ball_trajectory[-2]
        last_x, last_y = self.tennis_ball_trajectory[-1]
        return (last_x + (last_x - prev_x), last_y + (last_y - prev_y))

    def _append_valid_point(self, point):
        self.tennis_ball_trajectory.append(point)
        self.last_valid_position = point
        self.missing_frames = 0

    def _record_missing_detection(self):
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.last_valid_position = None

    def _mark_detection_rejected(self):
        self.last_detection["accepted"] = False
        self.last_detection["image"] = None

    def _empty_detection_state(self):
        return {
            "visible": False,
            "accepted": False,
            "image": None,
            "confidence": None,
            "candidate_count": 0,
        }

    def _distance(self, point_a, point_b):
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    def draw_trajectory(self, frame):
        if not self.tennis_ball_trajectory:
            return

        t0 = time.time()
        color = (87, 108, 255)
        points = list(self.tennis_ball_trajectory)

        for i, point in enumerate(points):
            radius = int(3 + (i / len(points)) * 4)
            cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)

        latest_point = points[-1]
        cv2.circle(frame, latest_point, 6, (0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing tennis ball trajectory took {time.time() - t0:.2f} sec")

    def handle_visualization(self, frame):
        if self.show_trajectory and self.tennis_ball_trajectory:
            self.draw_trajectory(frame)

    def clear_trajectory(self):
        self.tennis_ball_trajectory.clear()
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0

    def get_trajectory(self):
        return list(self.tennis_ball_trajectory)

    def get_last_detection(self):
        return dict(self.last_detection)

