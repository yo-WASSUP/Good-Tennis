import os
import tempfile
from tkinter import filedialog
import tkinter as tk
import time
import argparse


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtLineAutoDetector, CourtTrajectoryVisualizer, TennisBallTracker
    global BounceDetector, MiniMapVisualizer
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, YOLOPersonDetector, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "good-tennis-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .court.auto_detector import CourtLineAutoDetector as _CourtLineAutoDetector
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .analysis.bounce import BounceDetector as _BounceDetector
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .visualization.minimap import MiniMapVisualizer as _MiniMapVisualizer
        from .detection.tennis_ball import TennisBallTracker as _TennisBallTracker
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .detection.yolo_person import YOLOPersonDetector as _YOLOPersonDetector
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    CourtLineAutoDetector = _CourtLineAutoDetector
    PlayerTracker = _PlayerTracker
    BounceDetector = _BounceDetector
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    MiniMapVisualizer = _MiniMapVisualizer
    TennisBallTracker = _TennisBallTracker
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    YOLOPersonDetector = _YOLOPersonDetector
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION

class TennisAnalysisSystem:
    def __init__(self, video_path, show_display=True, 
                 show_skeletons=True, show_player_trajectories=True, 
                 show_court_trajectory=True, show_tennis_ball_trajectory=True,
                 show_player_stats=True, show_performance_stats=False, 
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/tennis-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='weights/yolo11s-pose.pt', player_detector='pose',
                 person_model='weights/yolo26s.pt', show_pose_roi=True,
                 court_detection='auto-fallback', show_bounce_detection=True,
                 bounce_classifier_path='', show_mini_map=True):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.ball_model_path = ball_model_path
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.player_detector = player_detector
        self.person_model = person_model
        self.show_pose_roi = show_pose_roi
        self.court_detection = court_detection
        self.bounce_classifier_path = bounce_classifier_path


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_tennis_ball_trajectory = show_tennis_ball_trajectory
        self.show_bounce_detection = show_bounce_detection
        self.show_mini_map = show_mini_map
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        if not os.path.exists(self.ball_model_path):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO tennis ball model and place it at "
                "weights/tennis-ball.pt, or pass its path with --ball-model."
            )
        
        self.person_detector = None
        if self.player_detector == 'yolo-person':
            self.rtmpose_processor = None
            self.person_detector = YOLOPersonDetector(model_path=self.person_model)
        elif self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(model_path=self.yolo_pose_model)
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)
        self.yolo_ball_model = YOLO(self.ball_model_path)

        self.last_stats_update_frame = 0


        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('outputs', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.bounce_events_path = os.path.join(self.save_dir, "bounce_events.json")
        self.cleaned_ball_trajectory_path = os.path.join(self.save_dir, "cleaned_ball_trajectory.json")
        self.output_video_path = os.path.join(self.save_dir, f"detect_{self.video_name}.mp4")
        self.detection_writer = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        self.tennis_ball_tracker = TennisBallTracker(
            yolo_ball_model=self.yolo_ball_model,
            trajectory_length=30,
            show_trajectory=False,
            show_performance_stats=self.show_performance_stats
        )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            person_detector=self.person_detector,
            player_detector=self.player_detector,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=self.show_performance_stats
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        self.minimap_visualizer = MiniMapVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
        self.bounce_detector = None
        self.court_detection_result = None
    def process_video(self):
        """Process the input video."""
        self.start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
        

        self.fps = fps
        

        template_path = self._get_template_path()
        template_gray, template_color = self._load_template(template_path, cap)
        

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self._setup_video_writer(self.frame_width, self.frame_height, fps)


        corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
        self.court_corners = corners
        self.court_roi_corners = roi_corners

        self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
        self.detection_writer = JsonlDetectionWriter(self.detections_path)
        

        self.court_mapper = CourtMapper(corners)
        self.player_pose_visualizer.court_mapper = self.court_mapper
        self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=30,
                                          detection_writer=self.detection_writer, fps=fps)
        self.bounce_detector = BounceDetector(fps=fps, classifier_path=self.bounce_classifier_path)
        

        self.stats_visualizer = StatsVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            language=self.language
        )
        
        frame_count = 0
        detect_frame_count = 0


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, detect_frame_count = self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)

        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        print(f"\n处理完成:")
        print(f"原始视频时长: {video_duration:.2f} 秒")
        print(f"处理耗时: {processing_time:.2f} 秒")
        print(f"处理速度比: {processing_time/video_duration:.2f}x")
        
        self._cleanup(cap)

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "tennis_ball": self.ball_model_path,
                "player_detector": self.player_detector,
                "person": self.person_model if self.player_detector == 'yolo-person' else None,
                "pose_family": self.pose_family if self.player_detector == 'pose' else None,
            },
            "analysis": {
                "court_detection": self.court_detection,
                "bounce_detection": self.show_bounce_detection,
                "bounce_method": "rule_lag20_postprocess" if not self.bounce_classifier_path else "clf_lag20_postprocess",
                "bounce_classifier": self.bounce_classifier_path,
                "mini_map": self.show_mini_map,
            },
            "court": {
                "template_path": template_path,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "detection_result": self.court_detection_result,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 10.97,
                    "length": 23.77,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
                "bounce_events": self.bounce_events_path,
                "cleaned_ball_trajectory": self.cleaned_ball_trajectory_path,
            },
        }
        write_json(self.metadata_path, metadata)

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray)
        
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0
            

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True

            self.rally_count += 1

            self.player_tracker.start_new_rally()
            

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False

            self.tennis_ball_tracker.clear_trajectory()
            if self.bounce_detector is not None:
                self.bounce_detector.clear()


        if not is_court:
            return frame, detect_frame_count

        detect_frame_count += 1

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        roi = frame[y1:y2, x1:x2]
        if self.show_pose_roi:
            cv2.rectangle(frame, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
            cv2.putText(frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        centroids, point_left_hands, point_right_hands = self.player_pose_visualizer.detect_players(roi, x1, y1)
        detected_ball_position = self.tennis_ball_tracker.detect_ball(frame, roi_corners=roi_corners)
        ball_position = self.tennis_ball_tracker.update_trajectory(detected_ball_position, roi_corners)
        ball_court_position = self.court_mapper.image_to_court(ball_position) if ball_position != [0, 0] else None
        bounce_event = None
        

        players = self.player_tracker.update(frame_count, centroids, ball_position, 
                                             point_left_hands, point_right_hands, detect_frame_count,
                                             ball_court_position=ball_court_position,
                                             bounce_event=bounce_event)
        

        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:

            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        t0 = time.time()

        self.player_pose_visualizer.draw_players(
            frame=frame, 
            player_tracker=self.player_tracker, 
            cached_movement_stats=self.cached_movement_stats,
            stats_visualizer=self.stats_visualizer if self.show_player_stats else None,
            rally_count=self.rally_count
        )
        t1 = time.time()
        if self.show_performance_stats:
            print(f"Drawing players took {t1 - t0:.2f} sec")
        

        if self.show_court_trajectory and not self.show_mini_map:
            t0 = time.time()
            frame = self.court_trajectory_visualizer.draw_overlay(frame, self.player_tracker.court_history)
            t1 = time.time()
            if self.show_performance_stats:
                print(f"Drawing court trajectory took {t1 - t0:.2f} sec")

        if self.show_mini_map:
            frame = self.minimap_visualizer.draw(
                frame,
                self.player_tracker.court_history,
                ball_court_position=None,
                bounce_events=[],
            )
        

        if frame is not None:
            if self.show_display:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            out.write(frame)

            if self.save_images:
                cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)
        return frame, detect_frame_count

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        
        return template_gray, template_color

    def _setup_video_writer(self, frame_width, frame_height, fps):

        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation."""

        if os.path.exists(os.path.join(self.save_dir, 'court_annotations.txt')):
            with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'r') as f:
                corners = eval(f.readline().split('=')[1])
                f.readline()
                mid_height = eval(f.readline().split('=')[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
            self.court_detection_result = {
                "status": "cached",
                "source": "court_annotations.txt",
            }
        else:
            corners, roi_corners, mid_height = self._detect_or_annotate_court(template_color)
       
        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError("Court annotation is incomplete: click 4 court corners in order. ROI is generated automatically.")

        with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        return corners, roi_corners, mid_height

    def _detect_or_annotate_court(self, template_color):
        if self.court_detection in ('auto', 'auto-fallback'):
            detector = CourtLineAutoDetector()
            detected = detector.detect(template_color)
            candidate = detected or self._candidate_from_diagnostics(template_color, detector.last_diagnostics)
            if detected:
                preview_path = self._write_auto_court_preview(template_color, detected)
                print(
                    "自动检测到球场线: "
                    f"confidence={detected['confidence']:.2f}, "
                    f"lines={detected['line_count']}, preview={preview_path}"
                )
                if self._confirm_auto_court_detection(template_color, detected):
                    self.court_detection_result = {
                        "status": "auto",
                        "accepted": True,
                        "confidence": detected["confidence"],
                        "preview": preview_path,
                        "diagnostics": detected.get("diagnostics"),
                    }
                    return detected["corners"], detected["roi_corners"], detected["mid_height"]

                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "confidence": detected["confidence"],
                    "preview": preview_path,
                    "diagnostics": detected.get("diagnostics"),
                }
                print("用户拒绝自动检测结果，切换到手动四角标注。")
            elif candidate:
                preview_path = self._write_auto_court_preview(template_color, candidate)
                print(
                    "自动检测置信度偏低，已显示候选球场线: "
                    f"confidence={candidate['confidence']:.2f}, "
                    f"lines={candidate['line_count']}, preview={preview_path}"
                )
                if self._confirm_auto_court_detection(template_color, candidate):
                    self.court_detection_result = {
                        "status": "auto_low_confidence_accepted",
                        "accepted": True,
                        "confidence": candidate["confidence"],
                        "preview": preview_path,
                        "diagnostics": candidate.get("diagnostics"),
                    }
                    return candidate["corners"], candidate["roi_corners"], candidate["mid_height"]

                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "confidence": candidate["confidence"],
                    "preview": preview_path,
                    "diagnostics": candidate.get("diagnostics"),
                }
                print("用户拒绝低置信度自动检测结果，切换到手动四角标注。")

            if self.court_detection == 'auto':
                if self.court_detection_result is None:
                    self.court_detection_result = {
                        "status": "auto",
                        "accepted": False,
                        "diagnostics": detector.last_diagnostics,
                    }
                raise RuntimeError(
                    "Court auto-detection failed: "
                    f"{detector.last_diagnostics}. Use --court-detection auto-fallback or manual."
                )

            if self.court_detection_result is None:
                print(f"自动球场线检测失败，切换到手动四角标注。diagnostics={detector.last_diagnostics}")
                self.court_detection_result = {
                    "status": "manual_fallback",
                    "accepted": False,
                    "diagnostics": detector.last_diagnostics,
                }

        corners, roi_corners, mid_height = annotate_court(template_color)
        if self.court_detection_result is None:
            self.court_detection_result = {
                "status": "manual",
                "accepted": True,
            }
        return corners, roi_corners, mid_height

    def _candidate_from_diagnostics(self, template_color, diagnostics):
        if not diagnostics or not diagnostics.get("corners"):
            return None

        corners = [(int(x), int(y)) for x, y in diagnostics["corners"]]
        roi_corners = compute_expanded_roi(corners, template_color.shape)
        mid_height = int((corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4)
        return {
            "corners": corners,
            "roi_corners": roi_corners,
            "mid_height": mid_height,
            "line_count": int(diagnostics.get("line_count", 0)),
            "confidence": float(diagnostics.get("confidence", 0.0)),
            "diagnostics": diagnostics,
        }

    def _confirm_auto_court_detection(self, template_color, detected):
        preview = self._build_auto_court_preview(template_color, detected)
        cv2.putText(
            preview,
            "Enter/Y: accept auto  M/R/Esc: manual corners",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        window_name = "Auto court detection preview"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, preview)
        print("请检查自动球场线预览：按 Enter/Y 接受，按 M/R/Esc 改为手动四角标注。")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10, ord('y'), ord('Y')):
                cv2.destroyWindow(window_name)
                return True
            if key in (27, ord('m'), ord('M'), ord('r'), ord('R')):
                cv2.destroyWindow(window_name)
                return False

    def _write_auto_court_preview(self, template_color, detected):
        preview = self._build_auto_court_preview(template_color, detected)
        preview_path = os.path.join(self.save_dir, "auto_court_preview.png")
        cv2.imwrite(preview_path, preview)
        return preview_path

    def _build_auto_court_preview(self, template_color, detected):
        preview = template_color.copy()
        corners = np.array(detected["corners"], dtype=np.int32)
        cv2.polylines(preview, [corners], True, (0, 255, 255), 3, cv2.LINE_AA)
        for index, point in enumerate(detected["corners"], start=1):
            cv2.circle(preview, tuple(point), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            preview,
            f"auto court confidence={detected['confidence']:.2f}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview

    def _cleanup(self, cap):
        """Clean up resources and merge audio when needed."""
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None

        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            time.sleep(1)

        cap.release()

        if self.show_bounce_detection and self.bounce_detector is not None:
            self._finalize_bounce_detection()

        if self.show_display:
            cv2.destroyAllWindows()

        if hasattr(self, 'keep_audio') and self.keep_audio:
            vap.process_video_with_audio(
                video_path=self.video_path,
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path,
                save_dir=self.save_dir
            )
        else:
            vap.process_video_without_audio(
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path
            )

    def _finalize_bounce_detection(self):
        if not os.path.exists(self.detections_path):
            return

        events = self.bounce_detector.process_detections(
            self.detections_path,
            output_path=self.bounce_events_path,
            trajectory_output_path=self.cleaned_ball_trajectory_path,
            rewrite_detections=True,
        )
        print(f"弹跳后处理完成: {len(events)} 个候选点，结果={self.bounce_events_path}")
        if not os.path.exists(self.temp_output_video_path):
            return

        annotated_path = os.path.join(self.save_dir, f"temp_bounce_{self.video_name}.mp4")
        self.bounce_detector.annotate_video(
            self.temp_output_video_path,
            annotated_path,
            events,
            trajectory_points=self.bounce_detector.processed_points,
            draw_minimap_bounces=self.show_mini_map,
            draw_processed_trajectory=self.show_tennis_ball_trajectory,
        )
        if os.path.exists(annotated_path):
            os.replace(annotated_path, self.temp_output_video_path)

    def analyze_tennis_ball(self, roi_corners, corners):
        """Hit-point analysis is currently disabled."""
        raise RuntimeError(
            "Hit-point analysis is disabled until it is migrated to detections.jsonl."
        )

    def is_court_view(self, frame, template_gray, threshold=0.75):
        """Return whether the frame matches the court template."""
        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        # print("match score: ", result)
        return np.max(result) >= threshold

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay


