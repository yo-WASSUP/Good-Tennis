import cv2
import numpy as np

from .mapper import compute_expanded_roi


class CourtLineAutoDetector:
    """Best-effort tennis court corner detector from visible court lines."""

    def __init__(self, min_line_length_ratio=0.16, min_confidence=0.72):
        self.min_line_length_ratio = min_line_length_ratio
        self.min_confidence = min_confidence
        self.last_diagnostics = {
            "status": "not_run",
            "reason": None,
            "confidence": 0.0,
        }

    def detect(self, image):
        if image is None or not isinstance(image, np.ndarray):
            self._set_failure("invalid_image")
            return None

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        mask = self._line_mask(gray)
        min_line_length = int(min(width, height) * self.min_line_length_ratio)
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=max(40, min_line_length),
            maxLineGap=max(12, min_line_length // 6),
        )
        if lines is None:
            self._set_failure("no_hough_lines")
            return None

        segments = [tuple(line[0]) for line in lines]
        horizontal, vertical = self._split_segments(segments, width, height)
        if len(horizontal) < 2 or len(vertical) < 2:
            self._set_failure("not_enough_horizontal_or_vertical_lines", line_count=len(segments), horizontal_count=len(horizontal), vertical_count=len(vertical))
            return None

        model = self._select_court_model(horizontal, vertical, width, height)
        if model is None:
            self._set_failure("no_consistent_court_model", line_count=len(segments), horizontal_count=len(horizontal), vertical_count=len(vertical))
            return None
        corners = model["corners"]

        corners = [(int(round(x)), int(round(y))) for x, y in corners]
        if not self._valid_corners(corners, width, height):
            self._set_failure("invalid_corner_geometry", corners=corners, line_count=len(segments), horizontal_count=len(horizontal), vertical_count=len(vertical))
            return None

        confidence, details = self._score_corners(corners, width, height, len(segments), len(horizontal), len(vertical))
        confidence = round(float(min(1.0, confidence + model["score_bonus"])), 4)
        details["model_score"] = round(float(model["score"]), 4)
        if confidence < self.min_confidence:
            self._set_failure("low_confidence", confidence=confidence, corners=corners, **details)
            return None

        roi_corners = compute_expanded_roi(corners, image.shape)
        mid_height = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4)
        self.last_diagnostics = {
            "status": "accepted",
            "reason": None,
            "confidence": confidence,
            "corners": corners,
            **details,
        }
        return {
            "corners": corners,
            "roi_corners": roi_corners,
            "mid_height": mid_height,
            "line_count": len(segments),
            "confidence": confidence,
            "diagnostics": dict(self.last_diagnostics),
        }

    def _line_mask(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bright = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(blurred, 50, 150)
        mask = cv2.bitwise_or(bright, edges)
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _split_segments(self, segments, width, height):
        horizontal = []
        vertical = []
        for x1, y1, x2, y2 in segments:
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            item = {
                "points": (float(x1), float(y1), float(x2), float(y2)),
                "mid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "length": length,
            }
            if self._is_frame_border_segment(item, width, height):
                continue
            if angle <= 14 or angle >= 166:
                if item["mid"][1] < height * 0.10 or item["mid"][1] > height * 0.94:
                    continue
                horizontal.append(item)
            elif 45 <= angle <= 135:
                vertical.append(item)

        horizontal.sort(key=lambda item: item["length"], reverse=True)
        vertical.sort(key=lambda item: item["length"], reverse=True)
        return horizontal[:18], vertical[:24]

    def _is_frame_border_segment(self, segment, width, height):
        x1, y1, x2, y2 = segment["points"]
        border = 3
        near_left = max(x1, x2) <= border
        near_right = min(x1, x2) >= width - 1 - border
        near_top = max(y1, y2) <= border
        near_bottom = min(y1, y2) >= height - 1 - border
        return near_left or near_right or near_top or near_bottom

    def _select_court_model(self, horizontal, vertical, width, height):
        best = None
        horizontal_pairs = []
        for top in horizontal:
            for bottom in horizontal:
                if bottom["mid"][1] <= top["mid"][1] + height * 0.22:
                    continue
                if top["mid"][1] < height * 0.12 or bottom["mid"][1] > height * 0.90:
                    continue
                top_span = self._x_span(top)
                bottom_span = self._x_span(bottom)
                if self._span_width(top_span) < width * 0.25 or self._span_width(bottom_span) < width * 0.35:
                    continue
                horizontal_pairs.append((top, bottom, top_span, bottom_span))

        for top, bottom, top_span, bottom_span in horizontal_pairs:
            for left_index, left in enumerate(vertical):
                left_top = self._intersection(left, top)
                left_bottom = self._intersection(left, bottom)
                if not self._valid_side_intersections(left_top, left_bottom, width, height):
                    continue

                for right in vertical[left_index + 1:]:
                    right_top = self._intersection(right, top)
                    right_bottom = self._intersection(right, bottom)
                    if not self._valid_side_intersections(right_top, right_bottom, width, height):
                        continue

                    corners = [left_top, right_top, right_bottom, left_bottom]
                    ordered = self._order_candidate_corners(corners)
                    if ordered is None:
                        continue
                    score = self._court_model_score(ordered, top_span, bottom_span, width, height)
                    if score <= 0:
                        continue
                    if best is None or score > best["score"]:
                        best = {
                            "corners": ordered,
                            "score": score,
                            "score_bonus": min(0.10, score * 0.08),
                        }
        return best

    def _x_span(self, line):
        x1, _, x2, _ = line["points"]
        return (min(x1, x2), max(x1, x2))

    def _span_width(self, span):
        return float(span[1] - span[0])

    def _valid_side_intersections(self, top_point, bottom_point, width, height):
        if top_point is None or bottom_point is None:
            return False
        tx, ty = top_point
        bx, by = bottom_point
        margin_x = width * 0.10
        margin_y = height * 0.08
        if not (-margin_x <= tx <= width + margin_x and -margin_x <= bx <= width + margin_x):
            return False
        if not (-margin_y <= ty <= height + margin_y and -margin_y <= by <= height + margin_y):
            return False
        return by > ty + height * 0.20

    def _order_candidate_corners(self, corners):
        points = sorted(corners, key=lambda point: point[1])
        top_points = sorted(points[:2], key=lambda point: point[0])
        bottom_points = sorted(points[2:], key=lambda point: point[0])
        ordered = [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
        if ordered[0][0] >= ordered[1][0] or ordered[3][0] >= ordered[2][0]:
            return None
        return ordered

    def _court_model_score(self, corners, top_span, bottom_span, width, height):
        points = np.array(corners, dtype=np.float32)
        area = abs(cv2.contourArea(points))
        if area < width * height * 0.14 or area > width * height * 0.82:
            return 0.0

        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        court_height = ((points[3][1] + points[2][1]) - (points[0][1] + points[1][1])) / 2.0
        if top_width < width * 0.22 or bottom_width < width * 0.35 or court_height < height * 0.36:
            return 0.0

        perspective_ratio = bottom_width / max(top_width, 1.0)
        if perspective_ratio < 1.05 or perspective_ratio > 2.80:
            return 0.0

        left_slant = points[3][0] - points[0][0]
        right_slant = points[2][0] - points[1][0]
        if left_slant > width * 0.08 or right_slant < -width * 0.08:
            return 0.0

        top_alignment = 1.0 - min(1.0, (
            abs(points[0][0] - top_span[0]) + abs(points[1][0] - top_span[1])
        ) / max(width * 0.45, 1.0))
        bottom_alignment = 1.0 - min(1.0, (
            abs(points[3][0] - bottom_span[0]) + abs(points[2][0] - bottom_span[1])
        ) / max(width * 0.45, 1.0))
        center_alignment = 1.0 - min(1.0, abs(float(np.mean(points[:, 0])) - width / 2.0) / (width * 0.36))
        border_penalty = 0.0
        if np.min(points[:, 1]) < height * 0.08 or np.max(points[:, 0]) > width * 0.96:
            border_penalty += 0.35
        if np.min(points[:, 0]) < width * 0.02 or np.max(points[:, 1]) > height * 0.94:
            border_penalty += 0.15

        size_score = min(1.0, (area / max(width * height, 1)) / 0.30)
        score = (
            0.24 * top_alignment
            + 0.20 * bottom_alignment
            + 0.20 * center_alignment
            + 0.20 * min(1.0, court_height / (height * 0.55))
            + 0.16 * size_score
            - border_penalty
        )
        return float(score)

    def _intersection(self, line_a, line_b):
        x1, y1, x2, y2 = line_a["points"]
        x3, y3, x4, y4 = line_b["points"]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    def _valid_corners(self, corners, width, height):
        points = np.array(corners, dtype=np.float32)
        margin_x = width * 0.08
        margin_y = height * 0.08
        if np.any(points[:, 0] < -margin_x) or np.any(points[:, 0] > width + margin_x):
            return False
        if np.any(points[:, 1] < -margin_y) or np.any(points[:, 1] > height + margin_y):
            return False

        area = abs(cv2.contourArea(points))
        frame_area = width * height
        if area < frame_area * 0.08 or area > frame_area * 0.95:
            return False

        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        left_height = np.linalg.norm(points[3] - points[0])
        right_height = np.linalg.norm(points[2] - points[1])
        return min(top_width, bottom_width, left_height, right_height) > min(width, height) * 0.08

    def _score_corners(self, corners, width, height, line_count, horizontal_count, vertical_count):
        points = np.array(corners, dtype=np.float32)
        area_ratio = abs(cv2.contourArea(points)) / max(1, width * height)
        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        left_height = np.linalg.norm(points[3] - points[0])
        right_height = np.linalg.norm(points[2] - points[1])

        width_balance = min(top_width, bottom_width) / max(top_width, bottom_width, 1.0)
        height_balance = min(left_height, right_height) / max(left_height, right_height, 1.0)
        line_support = min(1.0, (horizontal_count + vertical_count) / 12.0)
        area_score = 1.0 - min(1.0, abs(area_ratio - 0.45) / 0.45)

        border_distance = min(
            float(np.min(points[:, 0])),
            float(width - np.max(points[:, 0])),
            float(np.min(points[:, 1])),
            float(height - np.max(points[:, 1])),
        )
        border_score = min(1.0, max(0.0, border_distance / (min(width, height) * 0.035)))

        confidence = (
            0.24 * width_balance
            + 0.20 * height_balance
            + 0.18 * area_score
            + 0.18 * line_support
            + 0.20 * border_score
        )
        details = {
            "line_count": int(line_count),
            "horizontal_count": int(horizontal_count),
            "vertical_count": int(vertical_count),
            "area_ratio": round(float(area_ratio), 4),
            "width_balance": round(float(width_balance), 4),
            "height_balance": round(float(height_balance), 4),
            "border_score": round(float(border_score), 4),
        }
        return round(float(confidence), 4), details

    def _set_failure(self, reason, **details):
        self.last_diagnostics = {
            "status": "failed",
            "reason": reason,
            "confidence": float(details.pop("confidence", 0.0)),
            **details,
        }
