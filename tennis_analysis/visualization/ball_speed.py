"""Broadcast-style ball speed radar-gun overlay."""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class BallSpeedVisualizer:
    """Draw the latest measured shot speed as a radar-gun overlay.

    The overlay is drawn during the trajectory post-processing pass (alongside
    the cleaned ball trajectory and bounces), because the per-shot speed is
    only known after the full video has been analyzed.
    """

    def __init__(
        self,
        shots,
        speed_by_frame=None,
        frame_width=1920,
        frame_height=1080,
        language="en",
        hold_sec=3.0,
        fade_sec=2.0,
        min_alpha=0.45,
    ):
        self.shots = sorted(shots or [], key=lambda shot: int(shot.get("peak_frame", 0)))
        self.speed_by_frame = speed_by_frame or {}
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.language = language

        self.scale_factor = 2 * min(self.frame_width / 1920.0, self.frame_height / 1080.0)
        self.hold_frames = max(1, int(hold_sec * 30))
        self.fade_frames = max(1, int(fade_sec * 30))
        self.min_alpha = float(min_alpha)

        self.font_path = self._find_font()
        self._font_cache = {}
        self._peak_frames = [int(shot.get("peak_frame", 0)) for shot in self.shots]
        self._max_speed = max(
            (float(shot.get("peak_speed_kmh", 0)) for shot in self.shots),
            default=0.0,
        )

        self.texts = {
            "zh": {"label": "球速", "unit": "km/h", "max": "最大"},
            "en": {"label": "BALL SPEED", "unit": "km/h", "max": "MAX"},
        }

    def draw(self, frame, frame_index):
        if not self.shots:
            return frame
        shot = self._active_shot(int(frame_index))
        if shot is None:
            return frame
        age_frames = int(frame_index) - int(shot.get("peak_frame", 0))
        if age_frames < 0:
            return frame
        alpha = self._alpha(age_frames)
        if alpha <= 0.02:
            return frame
        self._draw_box(frame, shot, alpha)
        return frame

    def _active_shot(self, frame_index):
        # Latest shot whose speed has already been "measured" (peak reached).
        latest = None
        for shot, peak_frame in zip(self.shots, self._peak_frames):
            if peak_frame <= frame_index:
                latest = shot
            else:
                break
        return latest

    def _alpha(self, age_frames):
        if age_frames <= self.hold_frames:
            return 1.0
        if age_frames <= self.hold_frames + self.fade_frames:
            progress = (age_frames - self.hold_frames) / max(1, self.fade_frames)
            return 1.0 - (1.0 - self.min_alpha) * progress
        return self.min_alpha

    def _draw_box(self, frame, shot, alpha):
        box_w = max(160, int(250 * self.scale_factor))
        box_h = max(70, int(100 * self.scale_factor))
        x = (self.frame_width - box_w) // 2
        y = max(8, int(14 * self.scale_factor))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (12, 18, 30), -1)
        cv2.addWeighted(overlay, 0.62 * alpha + 0.18, frame, 1 - (0.62 * alpha + 0.18), 0, frame)
        accent = (0, 255, 200)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), accent, max(1, int(2 * self.scale_factor)), cv2.LINE_AA)
        cv2.line(frame, (x, y + max(1, int(30 * self.scale_factor))), (x + box_w, y + max(1, int(30 * self.scale_factor))), (60, 75, 90), 1, cv2.LINE_AA)

        texts = self.texts[self.language]
        speed = float(shot.get("peak_speed_kmh", 0))

        label_scale = max(0.4, 0.5 * self.scale_factor)
        number_scale = max(1.0, 1.5 * self.scale_factor)
        unit_scale = max(0.35, 0.45 * self.scale_factor)

        text_items = [
            (texts["label"], (x + int(12 * self.scale_factor), y + int(6 * self.scale_factor)), label_scale, (190, 230, 255), max(1, int(1 * self.scale_factor))),
            (f"{int(round(speed))}", (x + int(14 * self.scale_factor), y + int(30 * self.scale_factor)), number_scale, accent, max(1, int(2 * self.scale_factor))),
            (texts["unit"], (x + int(14 * self.scale_factor) + self._text_width(f"{int(round(speed))}", number_scale), y + int(46 * self.scale_factor)), unit_scale, (190, 230, 255), max(1, int(1 * self.scale_factor))),
        ]
        if self._max_speed > 0:
            max_text = f"{texts['max']} {int(round(self._max_speed))}"
            text_items.append((max_text, (x + box_w - int(12 * self.scale_factor) - self._text_width(max_text, unit_scale), y + int(6 * self.scale_factor)), unit_scale, (255, 220, 120), max(1, int(1 * self.scale_factor))))

        self._draw_text_items(frame, text_items, alpha)

    def _text_width(self, text, font_scale):
        if self.language == "en" or self.font_path is None:
            (width, _height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            return width
        # Measure with the same PIL font used for rendering so zh layout matches.
        font = self._get_font(font_scale)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    def _draw_text_items(self, frame, text_items, alpha):
        if self.language == "en" or self.font_path is None:
            for text, position, font_scale, color, thickness in text_items:
                color = self._fade(color, alpha)
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            return

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for text, position, font_scale, color, _thickness in text_items:
            faded = self._fade(color, alpha)
            font = self._get_font(font_scale)
            draw.text(position, text, font=font, fill=(faded[2], faded[1], faded[0]))
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fade(bgr_color, alpha):
        alpha = max(0.0, min(1.0, float(alpha)))
        return tuple(int(round(c * alpha + (1.0 - alpha) * 18)) for c in bgr_color)

    def _get_font(self, font_scale):
        font_size = max(10, int(font_scale * 30))
        if font_size not in self._font_cache:
            self._font_cache[font_size] = ImageFont.truetype(self.font_path, font_size)
        return self._font_cache[font_size]

    @staticmethod
    def _find_font():
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "simhei.ttf"),
            "simhei.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return None
