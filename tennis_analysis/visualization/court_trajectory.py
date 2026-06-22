import cv2
import numpy as np

from ..court.mapper import (
    TENNIS_COURT_LENGTH,
    TENNIS_DOUBLES_WIDTH,
    TENNIS_SERVICE_LINE_FROM_NET,
    TENNIS_SINGLES_WIDTH,
)


class CourtTrajectoryVisualizer:
    def __init__(self, width=200, height=430, margin_x=3.0, margin_y=5.0):
        self.default_width = width
        self.default_height = height
        self.width = width
        self.height = height
        self.doubles_width = TENNIS_DOUBLES_WIDTH
        self.court_length = TENNIS_COURT_LENGTH
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.court_overlay = self._create_court_overlay(self.width, self.height)

    def _get_view_transform(self, width, height):
        view_width = self.doubles_width + self.margin_x * 2
        view_length = self.court_length + self.margin_y * 2
        scale_x = (width - 20) / view_width
        scale_y = (height - 20) / view_length
        scale = min(scale_x, scale_y)
        offset_x = int((width - view_width * scale) / 2)
        offset_y = int((height - view_length * scale) / 2)
        return scale, offset_x, offset_y

    def _scale_point(self, x, y, scale, offset_x, offset_y):
        return int((x + self.margin_x) * scale + offset_x), int((y + self.margin_y) * scale + offset_y)

    def _create_court_overlay(self, width, height):
        court = np.zeros((height, width, 3), dtype=np.uint8)
        doubles_width = TENNIS_DOUBLES_WIDTH
        court_length = TENNIS_COURT_LENGTH
        singles_margin = (TENNIS_DOUBLES_WIDTH - TENNIS_SINGLES_WIDTH) / 2
        net_y = court_length / 2
        service_top = net_y - TENNIS_SERVICE_LINE_FROM_NET
        service_bottom = net_y + TENNIS_SERVICE_LINE_FROM_NET
        center_x = doubles_width / 2

        scale, offset_x, offset_y = self._get_view_transform(width, height)
        line_width_px = max(1, int(0.05 * scale))
        line_color = (70, 90, 70)
        boundary_color = (35, 55, 35)

        def point(x, y):
            return self._scale_point(x, y, scale, offset_x, offset_y)

        cv2.rectangle(court, point(-self.margin_x, -self.margin_y), point(doubles_width + self.margin_x, court_length + self.margin_y), boundary_color, 1)
        cv2.rectangle(court, point(0, 0), point(doubles_width, court_length), line_color, line_width_px)
        cv2.line(court, point(singles_margin, 0), point(singles_margin, court_length), line_color, line_width_px)
        cv2.line(court, point(doubles_width - singles_margin, 0), point(doubles_width - singles_margin, court_length), line_color, line_width_px)
        cv2.line(court, point(0, net_y), point(doubles_width, net_y), line_color, line_width_px)
        cv2.line(court, point(singles_margin, service_top), point(doubles_width - singles_margin, service_top), line_color, line_width_px)
        cv2.line(court, point(singles_margin, service_bottom), point(doubles_width - singles_margin, service_bottom), line_color, line_width_px)
        cv2.line(court, point(center_x, service_top), point(center_x, service_bottom), line_color, line_width_px)
        return court

    def draw_overlay(self, frame, court_history):
        try:
            frame_height, frame_width = frame.shape[:2]
            scale_factor = min(frame_width / 1920.0, frame_height / 1080.0) * 1.5
            court_overlay_width = int(self.default_width * scale_factor)
            court_overlay_height = int(self.default_height * scale_factor)

            if court_overlay_width != self.width or court_overlay_height != self.height:
                self.width = court_overlay_width
                self.height = court_overlay_height
                self.court_overlay = self._create_court_overlay(self.width, self.height)

            overlay = self.court_overlay.copy()
            height, width = overlay.shape[:2]
            scale, offset_x, offset_y = self._get_view_transform(width, height)

            for position, color in [('upper', (0, 255, 255)), ('lower', (255, 0, 255))]:
                history = list(court_history.get(position, []))
                for index, pos in enumerate(history):
                    if pos is None or len(pos) < 2:
                        continue
                    x, y = self._scale_point(pos[0], pos[1], scale, offset_x, offset_y)
                    if 0 <= x < width and 0 <= y < height:
                        radius_min = max(2, int(2 * scale_factor))
                        radius_max = max(3, int(5 * scale_factor))
                        radius = int(radius_min + (index / len(history)) * (radius_max - radius_min)) if len(history) > 1 else radius_min
                        cv2.circle(overlay, (x, y), radius, color, -1)

            h, w = overlay.shape[:2]
            padding = max(10, int(20 * scale_factor))
            if frame.shape[0] >= padding + h and frame.shape[1] >= padding + w:
                roi = frame[padding:padding + h, frame.shape[1] - w - padding:frame.shape[1] - padding]
                if roi.shape == overlay.shape:
                    cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)
                    frame[padding:padding + h, frame.shape[1] - w - padding:frame.shape[1] - padding] = roi
        except Exception as exc:
            print(f"Drawing court trajectory failed: {exc}")
        return frame
