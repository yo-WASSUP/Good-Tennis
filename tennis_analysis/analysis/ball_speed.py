"""Tennis ball speed measurement.

The ball speed is measured on the 2D court plane (10.97m x 23.77m) using the
cleaned, interpolated trajectory produced by :class:`BounceDetector`. For every
trajectory point a court-plane speed (km/h) is estimated with a centered finite
difference, and the trajectory is split into shots at bounces and rally gaps.
The peak speed of each shot is reported as that shot's "球速" (ball speed).

Caveat: a single camera only gives the top-down court projection, so the
measured speed is the horizontal speed and underestimates the true 3D ball
speed (the vertical component is lost). High-arcing shots are underestimated
more than flat drives.
"""

import numpy as np

MS_TO_KMH = 3.6
TENNIS_MAX_SERVE_KMH = 270.0


class BallSpeedAnalyzer:
    """Compute tennis ball speed (km/h) from the cleaned court-plane trajectory."""

    def __init__(
        self,
        fps=30.0,
        window_frames=5,
        max_segment_gap_sec=0.5,
        min_shot_frames=3,
        max_speed_kmh=TENNIS_MAX_SERVE_KMH,
        max_window_gap=2,
    ):
        self.fps = max(float(fps), 1.0)
        # Centered finite-difference half window (total window = 2*K+1 frames).
        self.half_window = max(1, int(window_frames) // 2)
        self.max_segment_gap_frames = max(2, int(self.fps * float(max_segment_gap_sec)))
        self.min_shot_frames = max(1, int(min_shot_frames))
        self.max_speed_kmh = float(max_speed_kmh)
        # Allow up to this many missing frames inside a finite-difference window.
        self.max_window_gap = int(max_window_gap)

    def analyze(self, trajectory_points, bounce_events=None):
        """Analyze trajectory points and return a serializable speed report.

        ``trajectory_points`` may be :class:`TrajectoryPoint` objects or dicts
        exposing ``frame`` and ``court`` (``[x_meters, y_meters]`` or ``None``).
        """
        points = self._normalize_points(trajectory_points)
        segments = self._segment(points, bounce_events or [])
        shots = self._build_shots(points, segments)
        summary = self._summarize(shots)
        speed_by_frame = {}
        for segment in segments:
            for index, speed in enumerate(segment["speeds"]):
                if speed is None:
                    continue
                frame = int(segment["points"][index]["frame"])
                speed_by_frame[frame] = round(float(speed), 2)
        return {
            "schema_version": "1.0",
            "fps": round(self.fps, 6),
            "unit": "km/h",
            "method": "court_plane_centered_difference",
            "note": (
                "Speed is the 2D court-plane (10.97m x 23.77m) projection speed "
                "estimated from the cleaned, interpolated ball trajectory. It "
                "underestimates true 3D ball speed because the vertical component "
                "is not captured; flat drives are more accurate than high lobs."
            ),
            "summary": summary,
            "shots": shots,
            "speed_by_frame": speed_by_frame,
        }

    def _normalize_points(self, trajectory_points):
        points = []
        for raw in trajectory_points or []:
            if raw is None:
                continue
            frame = self._attr(raw, "frame")
            court = self._attr(raw, "court")
            image = self._attr(raw, "image")
            try:
                frame = int(frame)
            except (TypeError, ValueError):
                continue
            points.append(
                {
                    "frame": frame,
                    "court": self._clean_court(court),
                    "image": self._clean_point(image),
                }
            )
        points.sort(key=lambda item: item["frame"])
        return points

    def _segment(self, points, bounce_events):
        bounce_frames = sorted(
            {
                int(event["frame"])
                for event in bounce_events or []
                if self._is_int_frame(event.get("frame") if isinstance(event, dict) else None)
            }
        )
        bounce_set = set(bounce_frames)

        segments = []
        current = []
        prev_frame = None
        for point in points:
            frame = point["frame"]
            is_bounce_point = frame in bounce_set
            boundary = False
            if prev_frame is not None:
                if frame - prev_frame > self.max_segment_gap_frames:
                    boundary = True
                elif any(prev_frame < bounce_frame < frame for bounce_frame in bounce_frames):
                    boundary = True
            if is_bounce_point:
                # The bounce frame is a direction discontinuity; end the current
                # flight and drop this point so it pollutes neither side.
                if current:
                    segments.append(current)
                    current = []
                prev_frame = frame
                continue
            if boundary and current:
                segments.append(current)
                current = []
            current.append(point)
            prev_frame = frame
        if current:
            segments.append(current)

        enriched = []
        for segment in segments:
            enriched.append({"points": segment, "speeds": self._segment_speeds(segment)})
        return enriched

    @staticmethod
    def _is_int_frame(value):
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False

    def _segment_speeds(self, segment_points):
        n = len(segment_points)
        speeds = [None] * n
        k = self.half_window
        for i in range(n):
            a = i - k
            b = i + k
            if a < 0 or b >= n:
                speeds[i] = self._edge_speed(segment_points, i, k)
                continue
            court_a = segment_points[a]["court"]
            court_b = segment_points[b]["court"]
            if court_a is None or court_b is None:
                continue
            frame_gap = int(segment_points[b]["frame"] - segment_points[a]["frame"])
            expected_gap = 2 * k
            if frame_gap <= 0 or frame_gap > expected_gap + self.max_window_gap:
                continue
            dt = frame_gap / self.fps
            if dt <= 0:
                continue
            distance = float(np.linalg.norm(np.asarray(court_b) - np.asarray(court_a)))
            speed_kmh = distance / dt * MS_TO_KMH
            if speed_kmh > self.max_speed_kmh:
                continue
            speeds[i] = float(speed_kmh)
        return speeds

    def _edge_speed(self, segment_points, index, k):
        """One-sided finite difference for the first/last frames of a segment."""
        n = len(segment_points)
        if index < k:
            a, b = index, min(n - 1, index + k)
        else:
            a, b = max(0, index - k), index
        if a == b:
            return None
        court_a = segment_points[a]["court"]
        court_b = segment_points[b]["court"]
        if court_a is None or court_b is None:
            return None
        frame_gap = int(segment_points[b]["frame"] - segment_points[a]["frame"])
        if frame_gap <= 0 or frame_gap > k + self.max_window_gap:
            return None
        dt = frame_gap / self.fps
        distance = float(np.linalg.norm(np.asarray(court_b) - np.asarray(court_a)))
        speed_kmh = distance / dt * MS_TO_KMH
        if speed_kmh > self.max_speed_kmh:
            return None
        return float(speed_kmh)

    def _build_shots(self, points, segments):
        shots = []
        shot_index = 0
        rally_index = 0
        point_index = 0
        for seg_number, segment in enumerate(segments):
            seg_points = segment["points"]
            speeds = segment["speeds"]
            if len(seg_points) < self.min_shot_frames:
                point_index += len(seg_points)
                continue
            valid = [(i, s) for i, s in enumerate(speeds) if s is not None]
            if not valid:
                point_index += len(seg_points)
                continue
            peak_local, peak_speed = max(valid, key=lambda item: item[1])
            peak_point = seg_points[peak_local]
            avg_speed = float(np.mean([s for _, s in valid]))
            # A segment gap (rally break) starts a new rally; bounces split shots.
            first_frame = seg_points[0]["frame"]
            prev_frame = points[point_index - 1]["frame"] if point_index > 0 else None
            if prev_frame is not None and first_frame - prev_frame > self.max_segment_gap_frames:
                rally_index += 1
            shots.append(
                {
                    "shot_index": shot_index,
                    "rally": rally_index,
                    "start_frame": int(seg_points[0]["frame"]),
                    "end_frame": int(seg_points[-1]["frame"]),
                    "peak_frame": int(peak_point["frame"]),
                    "peak_speed_kmh": round(float(peak_speed), 2),
                    "peak_speed_ms": round(float(peak_speed) / MS_TO_KMH, 2),
                    "avg_speed_kmh": round(float(avg_speed), 2),
                    "peak_court": peak_point["court"],
                    "peak_image": peak_point["image"],
                    "duration_sec": round(
                        (seg_points[-1]["frame"] - seg_points[0]["frame"]) / self.fps, 4
                    ),
                }
            )
            shot_index += 1
            point_index += len(seg_points)
        return shots

    def _summarize(self, shots):
        if not shots:
            return {
                "shot_count": 0,
                "max_speed_kmh": 0.0,
                "avg_shot_speed_kmh": 0.0,
            }
        peaks = [shot["peak_speed_kmh"] for shot in shots]
        return {
            "shot_count": len(shots),
            "max_speed_kmh": round(float(max(peaks)), 2),
            "avg_shot_speed_kmh": round(float(np.mean(peaks)), 2),
        }

    @staticmethod
    def _attr(obj, name):
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    @staticmethod
    def _clean_point(value):
        if value is None:
            return None
        try:
            x, y = float(value[0]), float(value[1])
        except (TypeError, IndexError, ValueError):
            return None
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return [round(x, 2), round(y, 2)]

    def _clean_court(self, value):
        point = self._clean_point(value)
        if point is None:
            return None
        return [float(point[0]), float(point[1])]
