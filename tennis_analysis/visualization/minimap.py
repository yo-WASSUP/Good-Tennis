import cv2
import numpy as np

from ..court.mapper import (
    TENNIS_COURT_LENGTH,
    TENNIS_DOUBLES_WIDTH,
    TENNIS_SERVICE_LINE_FROM_NET,
    TENNIS_SINGLES_WIDTH,
)


class MiniMapVisualizer:
    def __init__(self, width=210, height=420, margin=18, court_margin_x=3.0, court_margin_y=5.0):
        self.width = width
        self.height = height
        self.margin = margin
        self.padding = 16
        self.court_margin_x = court_margin_x
        self.court_margin_y = court_margin_y

    def draw(self, frame, court_history, ball_court_position=None, bounce_events=None):
        if frame is None:
            return frame

        overlay = frame.copy()
        x1, y1, x2, y2 = self._bounds(frame)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 28, 36), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
        self._draw_court(frame, x1, y1)
        self._draw_histories(frame, x1, y1, court_history)
        self._draw_bounces(frame, x1, y1, bounce_events or [])
        self._draw_ball(frame, x1, y1, ball_court_position)
        cv2.putText(frame, "Mini map", (x1 + 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return frame

    def draw_bounce_events(self, frame, bounce_events):
        if frame is None or not bounce_events:
            return frame
        x1, y1, _, _ = self._bounds(frame)
        self._draw_bounces(frame, x1, y1, bounce_events)
        return frame

    def draw_processed_ball(self, frame, ball_court_position):
        if frame is None or not self._valid_court_point(ball_court_position):
            return frame
        x1, y1, _, _ = self._bounds(frame)
        point = self._court_to_map(x1, y1, float(ball_court_position[0]), float(ball_court_position[1]))
        cv2.circle(frame, point, 5, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 8, (0, 215, 255), 1, cv2.LINE_AA)
        return frame

    def _bounds(self, frame):
        x1 = max(8, frame.shape[1] - self.width - self.margin)
        y1 = self.margin
        return x1, y1, x1 + self.width, y1 + self.height

    def _draw_court(self, frame, x, y):
        line_color = (210, 210, 210)
        view_tl = self._court_to_map(x, y, -self.court_margin_x, -self.court_margin_y)
        view_br = self._court_to_map(x, y, TENNIS_DOUBLES_WIDTH + self.court_margin_x, TENNIS_COURT_LENGTH + self.court_margin_y)
        cv2.rectangle(frame, view_tl, view_br, (70, 90, 90), 1, cv2.LINE_AA)

        tl = self._court_to_map(x, y, 0, 0)
        tr = self._court_to_map(x, y, TENNIS_DOUBLES_WIDTH, 0)
        br = self._court_to_map(x, y, TENNIS_DOUBLES_WIDTH, TENNIS_COURT_LENGTH)
        bl = self._court_to_map(x, y, 0, TENNIS_COURT_LENGTH)
        cv2.polylines(frame, [np.array([tl, tr, br, bl], dtype=np.int32)], True, line_color, 1, cv2.LINE_AA)

        singles_margin = (TENNIS_DOUBLES_WIDTH - TENNIS_SINGLES_WIDTH) / 2
        net_y = TENNIS_COURT_LENGTH / 2
        service_top = net_y - TENNIS_SERVICE_LINE_FROM_NET
        service_bottom = net_y + TENNIS_SERVICE_LINE_FROM_NET
        center_x = TENNIS_DOUBLES_WIDTH / 2

        lines = [
            ((singles_margin, 0), (singles_margin, TENNIS_COURT_LENGTH)),
            ((TENNIS_DOUBLES_WIDTH - singles_margin, 0), (TENNIS_DOUBLES_WIDTH - singles_margin, TENNIS_COURT_LENGTH)),
            ((0, net_y), (TENNIS_DOUBLES_WIDTH, net_y)),
            ((singles_margin, service_top), (TENNIS_DOUBLES_WIDTH - singles_margin, service_top)),
            ((singles_margin, service_bottom), (TENNIS_DOUBLES_WIDTH - singles_margin, service_bottom)),
            ((center_x, service_top), (center_x, service_bottom)),
        ]
        for start, end in lines:
            cv2.line(frame, self._court_to_map(x, y, *start), self._court_to_map(x, y, *end), line_color, 1, cv2.LINE_AA)

    def _draw_histories(self, frame, x, y, court_history):
        colors = {"upper": (80, 210, 255), "lower": (255, 150, 80)}
        for region, history in (court_history or {}).items():
            points = [self._court_to_map(x, y, float(p[0]), float(p[1])) for p in history if self._valid_court_point(p)]
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, colors.get(region, (255, 255, 255)), 1, cv2.LINE_AA)
            if points:
                cv2.circle(frame, points[-1], 4, colors.get(region, (255, 255, 255)), -1, cv2.LINE_AA)

    def _draw_ball(self, frame, x, y, ball_court_position):
        if not self._valid_court_point(ball_court_position):
            return
        point = self._court_to_map(x, y, float(ball_court_position[0]), float(ball_court_position[1]))
        cv2.circle(frame, point, 4, (0, 255, 255), -1, cv2.LINE_AA)

    def _draw_bounces(self, frame, x, y, bounce_events):
        for event in bounce_events[-12:]:
            court = event.get("court")
            if not self._valid_court_point(court):
                continue
            point = self._court_to_map(x, y, float(court[0]), float(court[1]))
            cv2.drawMarker(frame, point, (0, 215, 255), cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
            cv2.circle(frame, point, 5, (0, 215, 255), 1, cv2.LINE_AA)

    def _court_to_map(self, x, y, court_x, court_y):
        map_w = self.width - 2 * self.padding
        map_h = self.height - 2 * self.padding - 18
        view_width = TENNIS_DOUBLES_WIDTH + self.court_margin_x * 2
        view_length = TENNIS_COURT_LENGTH + self.court_margin_y * 2
        px = x + self.padding + int(((court_x + self.court_margin_x) / view_width) * map_w)
        py = y + self.padding + 20 + int(((court_y + self.court_margin_y) / view_length) * map_h)
        return px, py

    def _valid_court_point(self, point):
        if point is None:
            return False
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, IndexError, ValueError):
            return False
        return np.isfinite(x) and np.isfinite(y)
