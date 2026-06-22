import json
import os
import pickle
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..visualization.minimap import MiniMapVisualizer


class LegacyColumnConcatenator(BaseEstimator, TransformerMixin):
    """Compatibility shim for old sktime ColumnConcatenator pickles."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rows = []
        for _, row in X.iterrows():
            values = []
            for value in row:
                if isinstance(value, pd.Series):
                    values.extend(value.tolist())
                elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    values.extend(list(value))
                else:
                    values.append(value)
            rows.append(pd.Series(values))
        return pd.DataFrame({"ts": rows})


@dataclass
class TrajectoryPoint:
    frame: int
    time_sec: float
    image: list | None
    court: list | None
    interpolated: bool = False


class BounceDetector:
    """Post-process ball trajectory with a 20-frame x/y/velocity window."""

    def __init__(
        self,
        fps=30,
        window_size=20,
        center_offset=10,
        min_event_gap_sec=0.45,
        min_score=0.34,
        max_interpolation_gap=12,
        classifier_path="",
        court_margin=0.9,
        max_center_velocity=2500,
        max_speed_ratio=12,
    ):
        self.fps = max(float(fps), 1.0)
        self.window_size = int(window_size)
        self.center_offset = int(center_offset)
        self.min_event_gap_frames = max(1, int(float(min_event_gap_sec) * self.fps))
        self.min_score = float(min_score)
        self.max_interpolation_gap = int(max_interpolation_gap)
        self.classifier_path = classifier_path
        self.court_margin = float(court_margin)
        self.max_center_velocity = float(max_center_velocity)
        self.max_speed_ratio = float(max_speed_ratio)
        self.classifier = None
        self.classifier_error = None
        self.events = []
        self.processed_points = []

    def process_detections(self, detections_path, output_path=None, trajectory_output_path=None, rewrite_detections=True):
        records = self._load_records(detections_path)
        points = self._records_to_points(records)
        cleaned = self._remove_outliers(points)
        interpolated = self._interpolate(cleaned)
        self.processed_points = interpolated
        events = self.detect_from_points(interpolated)
        self.events = events

        if output_path:
            self._write_events(output_path, events)
        if trajectory_output_path:
            self._write_trajectory(trajectory_output_path, interpolated)
        if rewrite_detections:
            self._rewrite_records_with_bounces(detections_path, records, events)
        return events

    def detect_from_points(self, points):
        coords = np.array(
            [
                point.image if point.image is not None else [np.nan, np.nan]
                for point in points
            ],
            dtype=np.float32,
        )
        velocity = self._velocity(coords)
        raw_events = []
        classifier = self._load_classifier()
        if classifier is not None:
            return self._detect_with_classifier(classifier, points, coords, velocity)

        court_coords = np.array(
            [
                point.court if point.court is not None else [np.nan, np.nan]
                for point in points
            ],
            dtype=np.float32,
        )

        for end_index in range(self.window_size - 1, len(points)):
            start_index = end_index - self.window_size + 1
            center_index = end_index - self.center_offset
            if center_index <= 0 or center_index >= len(points) - 1:
                continue

            window = coords[start_index:end_index + 1]
            window_v = velocity[start_index:end_index + 1]
            if np.isnan(window).any() or np.isnan(window_v).any():
                continue

            court_window = court_coords[start_index:end_index + 1]
            if np.isnan(court_window).any():
                court_window = None
            score, diagnostics = self._score_window(window, window_v, self.window_size - self.center_offset - 1, court_window=court_window)
            is_bounce = score >= self.min_score
            if not is_bounce:
                continue

            point = points[center_index]
            if not self._valid_bounce_court_position(point.court):
                continue
            raw_events.append(
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)],
                    "court": point.court,
                    "confidence": round(float(score), 3),
                    "method": "clf_lag20" if classifier is not None else "trajectory_lag20",
                    "diagnostics": diagnostics,
                }
            )

        return self._dedupe_events(raw_events)

    def _detect_with_classifier(self, classifier, points, coords, velocity):
        feature_rows = []
        center_indices = []
        for row_index in range(self.window_size, len(points) - 1):
            window = coords[row_index - self.window_size:row_index]
            window_v = velocity[row_index - self.window_size:row_index]
            if np.isnan(window).any() or np.isnan(window_v).any():
                continue
            feature_rows.append(self._window_to_feature_row(window, window_v))
            center_indices.append(row_index - self.center_offset)

        if not feature_rows:
            return []

        features = pd.DataFrame(feature_rows)
        predictions = classifier.predict(features)
        probabilities = None
        if hasattr(classifier, "predict_proba"):
            try:
                probabilities = classifier.predict_proba(features)
            except Exception:
                probabilities = None

        classes = list(getattr(classifier, "classes_", []))
        raw_events = []
        for row_number, prediction in enumerate(predictions):
            if int(prediction) != 1:
                continue
            center_index = center_indices[row_number]
            if center_index < 0 or center_index >= len(points):
                continue
            point = points[center_index]
            if point.image is None:
                continue
            confidence = 1.0
            if probabilities is not None:
                if 1 in classes:
                    confidence = float(probabilities[row_number][classes.index(1)])
                elif len(probabilities[row_number]) > 1:
                    confidence = float(probabilities[row_number][-1])
            raw_events.append(
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)],
                    "court": point.court,
                    "confidence": round(float(confidence), 3),
                    "method": "clf_lag20",
                    "diagnostics": {
                        "classifier_path": self.classifier_path,
                        "prediction": int(prediction),
                        "window_size": int(self.window_size),
                        "lag_order": "20_to_1",
                    },
                }
            )
        return self._dedupe_events(raw_events)

    def annotate_video(
        self,
        input_video_path,
        output_video_path,
        events,
        trajectory_points=None,
        display_sec=0.45,
        trajectory_length=30,
        draw_minimap_bounces=True,
        draw_processed_trajectory=True,
    ):
        if not events and not trajectory_points:
            return False

        events = sorted(events, key=lambda event: int(event["frame"]))
        trajectory_points = trajectory_points or self.processed_points
        trajectory_by_frame = self._trajectory_by_frame(trajectory_points)
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video for bounce annotation: {input_video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        display_frames = max(1, int((fps or self.fps) * float(display_sec)))
        minimap = MiniMapVisualizer() if draw_minimap_bounces else None
        frame_index = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_index += 1
            active_events = [
                event for event in events
                if 0 <= frame_index - int(event["frame"]) <= display_frames
            ]
            if draw_processed_trajectory:
                self.draw_processed_trajectory(frame, frame_index, trajectory_by_frame, trajectory_length=trajectory_length)
            for event in active_events:
                self.draw_event(frame, event, age_frames=frame_index - int(event["frame"]), display_frames=display_frames)
            if minimap is not None and active_events:
                minimap.draw_bounce_events(frame, active_events)
            if minimap is not None and draw_processed_trajectory:
                current_point = trajectory_by_frame.get(frame_index)
                current_court = current_point.court if current_point is not None else None
                minimap.draw_processed_ball(frame, current_court)
            writer.write(frame)

        video.release()
        writer.release()
        return True

    def draw_processed_trajectory(self, frame, frame_index, trajectory_by_frame, trajectory_length=30):
        points = []
        start_frame = max(1, frame_index - int(trajectory_length) + 1)
        for candidate_frame in range(start_frame, frame_index + 1):
            point = trajectory_by_frame.get(candidate_frame)
            if point is not None and point.image is not None:
                points.append(point)
        if not points:
            return

        for index, point in enumerate(points):
            x, y = int(point.image[0]), int(point.image[1])
            age_ratio = (index + 1) / len(points)
            color = (0, int(120 + 95 * age_ratio), 255)
            radius = max(2, int(2 + 5 * age_ratio))
            thickness = -1 if not point.interpolated else 1
            cv2.circle(frame, (x, y), radius, color, thickness, cv2.LINE_AA)

        latest = points[-1]
        x, y = int(latest.image[0]), int(latest.image[1])
        cv2.circle(frame, (x, y), 7, (0, 215, 255), -1, cv2.LINE_AA)

    def draw_event(self, frame, event, age_frames=0, display_frames=1):
        image = event.get("image")
        if not image:
            return
        x, y = int(image[0]), int(image[1])
        progress = min(1.0, max(0.0, float(age_frames) / max(1.0, float(display_frames))))
        color = (0, 215, 255)
        radius = int(14 + 8 * progress)
        thickness = max(2, int(4 - 2 * progress))
        cv2.circle(frame, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Bounce {event.get('confidence', 0):.2f}",
            (x + 14, max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    def get_events(self):
        return list(self.events)

    def clear(self):
        self.events = []
        self.processed_points = []

    def _load_records(self, path):
        records = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _records_to_points(self, records):
        points = []
        for record in records:
            ball = record.get("tennis_ball") or {}
            points.append(
                TrajectoryPoint(
                    frame=int(record.get("frame", len(points) + 1)),
                    time_sec=float(record.get("time_sec") or 0.0),
                    image=self._point_or_none(ball.get("image")),
                    court=self._point_or_none(ball.get("court")),
                )
            )
        return points

    def _remove_outliers(self, points):
        cleaned = [TrajectoryPoint(**point.__dict__) for point in points]
        coords = np.array(
            [point.image if point.image is not None else [np.nan, np.nan] for point in cleaned],
            dtype=np.float32,
        )
        valid_indices = np.where(~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1]))[0]
        if len(valid_indices) < 5:
            return cleaned

        steps = []
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            frame_gap = max(1, cleaned[right].frame - cleaned[left].frame)
            steps.append(float(np.linalg.norm(coords[right] - coords[left]) / frame_gap))
        threshold = self._robust_threshold(np.array(steps, dtype=np.float32), floor=90.0)

        for index in valid_indices[1:-1]:
            prev_index = self._previous_valid(coords, index)
            next_index = self._next_valid(coords, index)
            if prev_index is None or next_index is None:
                continue
            prev_dist = float(np.linalg.norm(coords[index] - coords[prev_index]) / max(1, index - prev_index))
            next_dist = float(np.linalg.norm(coords[next_index] - coords[index]) / max(1, next_index - index))
            bridge_dist = float(np.linalg.norm(coords[next_index] - coords[prev_index]) / max(1, next_index - prev_index))
            isolated_jump = prev_dist > threshold and next_dist > threshold and bridge_dist < threshold
            if isolated_jump:
                cleaned[index].image = None
                cleaned[index].court = None
        return cleaned

    def _interpolate(self, points):
        interpolated = [TrajectoryPoint(**point.__dict__) for point in points]
        valid = [index for index, point in enumerate(interpolated) if point.image is not None]
        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left
            if gap <= 1 or gap - 1 > self.max_interpolation_gap:
                continue
            left_point = interpolated[left]
            right_point = interpolated[right]
            for index in range(left + 1, right):
                alpha = (index - left) / gap
                image = self._lerp(left_point.image, right_point.image, alpha)
                court = None
                if left_point.court is not None and right_point.court is not None:
                    court = self._lerp(left_point.court, right_point.court, alpha)
                interpolated[index].image = image
                interpolated[index].court = court
                interpolated[index].interpolated = True
        return interpolated

    def _velocity(self, coords):
        velocity = np.full(len(coords), np.nan, dtype=np.float32)
        for index in range(1, len(coords)):
            if np.isnan(coords[index]).any() or np.isnan(coords[index - 1]).any():
                continue
            velocity[index] = float(np.linalg.norm(coords[index] - coords[index - 1]) * self.fps)
        if len(velocity) > 1:
            velocity[0] = velocity[1]
        return velocity

    def _score_window(self, window, velocity, center, court_window=None):
        centered = window - np.nanmean(window, axis=0)
        scale = max(float(np.nanstd(centered)), 1.0)
        normalized = centered / scale
        smooth = self._smooth(window)
        center_point = smooth[center]
        before = smooth[max(0, center - 5):center]
        after = smooth[center + 1:min(len(smooth), center + 6)]
        if len(before) < 3 or len(after) < 3:
            return 0.0, {}

        before_center = np.mean(before, axis=0)
        after_center = np.mean(after, axis=0)
        v_in = center_point - before_center
        v_out = after_center - center_point
        turn_degrees = self._angle_between(v_in, v_out)
        deviation = self._point_line_distance(center_point, before_center, after_center)

        v_center = float(velocity[center])
        local_v = velocity[max(0, center - 4):min(len(velocity), center + 5)]
        median_v = float(np.nanmedian(local_v))
        peak_v = float(np.nanmax(local_v))
        speed_ratio = peak_v / max(median_v, 1.0)
        if v_center > self.max_center_velocity or speed_ratio > self.max_speed_ratio:
            return 0.0, {
                "reject_reason": "unstable_velocity",
                "center_velocity": round(float(v_center), 3),
                "speed_ratio": round(float(speed_ratio), 3),
                "window_size": int(self.window_size),
            }

        y = normalized[:, 1]
        y_slope_in = self._line_slope(np.arange(center + 1), y[:center + 1])
        y_slope_out = self._line_slope(np.arange(len(y) - center), y[center:])
        y_reversal = y_slope_in > 0.05 and y_slope_out < -0.05
        local_y_peak = window[center, 1] >= np.max(window[max(0, center - 5):min(len(window), center + 6), 1]) - 4.0
        local_y_valley = window[center, 1] <= np.min(window[max(0, center - 5):min(len(window), center + 6), 1]) + 4.0
        y_extreme = local_y_peak or local_y_valley

        court_turn = 0.0
        court_deviation = 0.0
        if court_window is not None:
            court_smooth = self._smooth(court_window)
            court_center = court_smooth[center]
            court_before = np.mean(court_smooth[max(0, center - 5):center], axis=0)
            court_after = np.mean(court_smooth[center + 1:min(len(court_smooth), center + 6)], axis=0)
            court_turn = self._angle_between(court_center - court_before, court_after - court_center)
            court_deviation = self._point_line_distance(court_center, court_before, court_after)

        angle_score = min(1.0, turn_degrees / 95.0)
        deviation_score = min(1.0, deviation / 18.0)
        speed_score = min(1.0, max(0.0, speed_ratio - 1.0) / 2.0)
        reversal_score = 1.0 if y_reversal else 0.0
        extreme_score = 1.0 if y_extreme else 0.0
        court_score = max(min(1.0, court_turn / 75.0), min(1.0, court_deviation / 0.55))
        if not (y_reversal or y_extreme):
            return 0.0, {
                "reject_reason": "no_local_y_extreme",
                "turn_degrees": round(float(turn_degrees), 3),
                "deviation_px": round(float(deviation), 3),
                "window_size": int(self.window_size),
            }
        score = (
            0.28 * angle_score
            + 0.24 * deviation_score
            + 0.12 * speed_score
            + 0.16 * reversal_score
            + 0.10 * extreme_score
            + 0.10 * court_score
        )

        diagnostics = {
            "turn_degrees": round(float(turn_degrees), 3),
            "deviation_px": round(float(deviation), 3),
            "center_velocity": round(float(v_center), 3),
            "speed_ratio": round(float(speed_ratio), 3),
            "y_slope_in": round(float(y_slope_in), 4),
            "y_slope_out": round(float(y_slope_out), 4),
            "local_y_peak": bool(local_y_peak),
            "local_y_valley": bool(local_y_valley),
            "court_turn_degrees": round(float(court_turn), 3),
            "court_deviation_m": round(float(court_deviation), 3),
            "window_size": int(self.window_size),
        }
        return float(score), diagnostics

    def _valid_bounce_court_position(self, court):
        if court is None:
            return True
        try:
            x, y = float(court[0]), float(court[1])
        except (TypeError, IndexError, ValueError):
            return False
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        return (
            -self.court_margin <= x <= 10.97 + self.court_margin
            and -self.court_margin <= y <= 23.77 + self.court_margin
        )

    def _load_classifier(self):
        if self.classifier is not None or self.classifier_error is not None:
            return self.classifier
        if not self.classifier_path or not os.path.exists(self.classifier_path):
            self.classifier_error = f"classifier not found: {self.classifier_path}"
            return None
        try:
            self._install_pickle_compat()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(self.classifier_path, "rb") as file:
                    self.classifier = pickle.load(file)
            self._repair_legacy_classifier(self.classifier)
        except Exception as exc:
            self.classifier_error = f"{type(exc).__name__}: {exc}"
            print(f"弹跳分类模型加载失败，回退规则评分: {self.classifier_error}")
            return None
        print(f"已加载弹跳分类模型: {self.classifier_path}")
        return self.classifier

    def _repair_legacy_classifier(self, classifier):
        estimators = []
        if hasattr(classifier, "steps"):
            estimators.extend(step for _, step in classifier.steps)
        estimators.append(classifier)
        for estimator in estimators:
            if not hasattr(estimator, "_is_vectorized"):
                estimator._is_vectorized = False
            if not hasattr(estimator, "_class_dictionary") and hasattr(estimator, "classes_"):
                estimator._class_dictionary = {
                    class_label: index for index, class_label in enumerate(list(estimator.classes_))
                }
            if not hasattr(estimator, "_y_metadata"):
                estimator._y_metadata = {"is_univariate": True}

    def _install_pickle_compat(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import sktime.transformations.panel.compose as panel_compose

            if not hasattr(panel_compose, "ColumnConcatenator"):
                panel_compose.ColumnConcatenator = LegacyColumnConcatenator
        except Exception:
            return

    def _window_to_feature_row(self, window, velocity):
        return {
            "x": pd.Series([float(window[index, 0]) for index in range(self.window_size)]),
            "y": pd.Series([float(window[index, 1]) for index in range(self.window_size)]),
            "V": pd.Series([float(velocity[index]) for index in range(self.window_size)]),
        }

    def _dedupe_events(self, events):
        selected = []
        for event in sorted(events, key=lambda item: item["confidence"], reverse=True):
            if any(abs(event["frame"] - kept["frame"]) < self.min_event_gap_frames for kept in selected):
                continue
            selected.append(event)
        return sorted(selected, key=lambda item: item["frame"])

    def _write_events(self, path, events):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"events": events}, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _write_trajectory(self, path, points):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "points": [
                {
                    "frame": int(point.frame),
                    "time_sec": round(float(point.time_sec), 6),
                    "image": [round(float(point.image[0]), 2), round(float(point.image[1]), 2)] if point.image is not None else None,
                    "court": [round(float(point.court[0]), 4), round(float(point.court[1]), 4)] if point.court is not None else None,
                    "interpolated": bool(point.interpolated),
                }
                for point in points
            ]
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")

    def _rewrite_records_with_bounces(self, path, records, events):
        by_frame = {int(event["frame"]): event for event in events}
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            for record in records:
                record["bounce"] = by_frame.get(int(record.get("frame", -1)))
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
        os.replace(tmp_path, path)

    def _point_or_none(self, point):
        if point is None:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, IndexError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return [x, y]

    def _trajectory_by_frame(self, points):
        return {int(point.frame): point for point in points or [] if point.image is not None}

    def _previous_valid(self, coords, index):
        for candidate in range(index - 1, -1, -1):
            if not np.isnan(coords[candidate]).any():
                return candidate
        return None

    def _next_valid(self, coords, index):
        for candidate in range(index + 1, len(coords)):
            if not np.isnan(coords[candidate]).any():
                return candidate
        return None

    def _robust_threshold(self, values, floor):
        if len(values) == 0:
            return floor
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 6.0 * max(mad, 1.0))

    def _smooth(self, points):
        if len(points) < 3:
            return points
        smoothed = points.copy()
        for index in range(1, len(points) - 1):
            smoothed[index] = (points[index - 1] + points[index] * 2 + points[index + 1]) / 4.0
        return smoothed

    def _lerp(self, start, end, alpha):
        return [
            float(start[0] + (end[0] - start[0]) * alpha),
            float(start[1] + (end[1] - start[1]) * alpha),
        ]

    def _line_slope(self, x, y):
        if len(x) < 2:
            return 0.0
        return float(np.polyfit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), 1)[0])

    def _angle_between(self, vec_a, vec_b):
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom <= 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _point_line_distance(self, point, line_start, line_end):
        line = line_end - line_start
        denom = float(np.linalg.norm(line))
        if denom <= 1e-6:
            return float(np.linalg.norm(point - line_start))
        return float(abs(np.cross(line, point - line_start)) / denom)
