"""Unit tests for BallSpeedAnalyzer (numpy-only, no CV/torch stack needed).

Run with:  python tests/test_ball_speed.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_analysis.analysis.ball_speed import BallSpeedAnalyzer

FPS = 30.0


def _linear_flight(start_frame, count, start_court, velocity_ms, fps=FPS):
    """Build a contiguous flight of points moving at a constant velocity."""
    direction = np.array([1.0, 0.0])
    points = []
    for i in range(count):
        frame = start_frame + i
        court = (np.array(start_court, dtype=float) + direction * velocity_ms * i / fps).tolist()
        points.append({"frame": frame, "court": court, "image": [100 + i, 100], "time_sec": frame / fps})
    return points


def _kmh_to_ms(kmh):
    return kmh / 3.6


def test_constant_speed():
    points = _linear_flight(start_frame=1, count=20, start_court=[2.0, 11.885], velocity_ms=_kmh_to_ms(100.0))
    analyzer = BallSpeedAnalyzer(fps=FPS)
    result = analyzer.analyze(points, bounce_events=[])
    assert result["summary"]["shot_count"] == 1, result["summary"]
    shot = result["shots"][0]
    assert abs(shot["peak_speed_kmh"] - 100.0) < 1.5, shot
    assert abs(shot["avg_speed_kmh"] - 100.0) < 1.5, shot
    print(f"  constant speed: peak={shot['peak_speed_kmh']:.2f} km/h (expected ~100)")


def test_two_shots_split_by_bounce():
    shot1 = _linear_flight(start_frame=1, count=12, start_court=[1.0, 11.885], velocity_ms=_kmh_to_ms(100.0))
    bounce_frame = shot1[-1]["frame"] + 1
    shot2_start = bounce_frame + 1
    # Returning shot moving the opposite direction (negative x), faster.
    shot2 = []
    direction = np.array([-1.0, 0.0])
    for i in range(12):
        frame = shot2_start + i
        court = (np.array([9.0, 11.885], dtype=float) + direction * _kmh_to_ms(130.0) * i / FPS).tolist()
        shot2.append({"frame": frame, "court": court, "image": [100, 100], "time_sec": frame / FPS})

    analyzer = BallSpeedAnalyzer(fps=FPS)
    result = analyzer.analyze(shot1 + shot2, bounce_events=[{"frame": bounce_frame}])
    assert result["summary"]["shot_count"] == 2, result["summary"]
    peaks = sorted(shot["peak_speed_kmh"] for shot in result["shots"])
    assert abs(peaks[0] - 100.0) < 2.0, peaks
    assert abs(peaks[1] - 130.0) < 2.0, peaks
    print(f"  two shots: peaks={peaks} (expected ~[100, 130])")


def test_empty_and_degenerate():
    analyzer = BallSpeedAnalyzer(fps=FPS)
    assert analyzer.analyze([], [])["summary"]["shot_count"] == 0
    assert analyzer.analyze([{"frame": 1, "court": None, "image": None}], [])["summary"]["shot_count"] == 0
    # Two points: shorter than min_shot_frames default (3) -> no shot.
    two = _linear_flight(start_frame=1, count=2, start_court=[2.0, 11.885], velocity_ms=_kmh_to_ms(100.0))
    assert analyzer.analyze(two, [])["summary"]["shot_count"] == 0
    print("  empty/degenerate: ok")


def test_outlier_rejected():
    points = _linear_flight(start_frame=1, count=20, start_court=[2.0, 11.885], velocity_ms=_kmh_to_ms(100.0))
    # Inject a single-frame teleport that would imply an impossible speed.
    points[10] = {"frame": points[10]["frame"], "court": [200.0, 11.885], "image": [0, 0]}
    analyzer = BallSpeedAnalyzer(fps=FPS)
    result = analyzer.analyze(points, bounce_events=[])
    # Peak must stay below the physical cap; the outlier must not become the peak.
    assert result["summary"]["max_speed_kmh"] <= analyzer.max_speed_kmh + 0.1
    print(f"  outlier rejected: max={result['summary']['max_speed_kmh']:.2f} km/h")


def test_rally_gap_splits_rallies():
    rally1 = _linear_flight(start_frame=1, count=10, start_court=[1.0, 11.885], velocity_ms=_kmh_to_ms(100.0))
    gap = 60  # > 0.5s -> new rally
    rally2 = _linear_flight(start_frame=rally1[-1]["frame"] + gap, count=10, start_court=[1.0, 11.885], velocity_ms=_kmh_to_ms(120.0))
    analyzer = BallSpeedAnalyzer(fps=FPS)
    result = analyzer.analyze(rally1 + rally2, bounce_events=[])
    rallies = {shot["rally"] for shot in result["shots"]}
    assert rallies == {0, 1}, rallies
    print(f"  rally gap splits: rallies={sorted(rallies)}")


def test_output_frame_indices_track_compacted_position():
    # Simulate non-court frames being skipped: original frames 1,2,3 then 10,11,12
    # (4-9 skipped). The cleaned trajectory has 6 points, so output frames are 1..6
    # even though original frames are 1,2,3,10,11,12.
    frames = [1, 2, 3, 10, 11, 12]
    points = []
    for i, f in enumerate(frames):
        court = [2.0 + 27.78 * i / FPS, 11.885]  # constant ~100 km/h
        points.append({"frame": f, "court": court, "image": [100 + i, 100]})
    analyzer = BallSpeedAnalyzer(fps=FPS)
    result = analyzer.analyze(points, bounce_events=[])
    assert result["summary"]["shot_count"] == 1
    shot = result["shots"][0]
    # Original frames keep the gappy input numbering.
    assert shot["start_frame"] == 1, shot
    assert shot["end_frame"] == 12, shot
    # Output frames are the compacted 1-based trajectory position.
    assert shot["start_output_frame"] == 1, shot
    assert shot["end_output_frame"] == 6, shot
    assert 1 <= shot["peak_output_frame"] <= 6, shot
    print(
        f"  output-frame indices: start_frame={shot['start_frame']} end_frame={shot['end_frame']} "
        f"| start_output_frame={shot['start_output_frame']} end_output_frame={shot['end_output_frame']} "
        f"peak_output_frame={shot['peak_output_frame']}"
    )


def test_visualizer_aligns_to_output_frames():
    # One shot whose peak is at output frame 4 but original frame 10 (skipped
    # non-court frames 4-9). The overlay must activate at output frame 4, not 10.
    from tennis_analysis.visualization.ball_speed import BallSpeedVisualizer

    shot = {
        "shot_index": 0, "rally": 0,
        "start_frame": 1, "end_frame": 12, "peak_frame": 10,
        "start_output_frame": 1, "end_output_frame": 6, "peak_output_frame": 4,
        "peak_speed_kmh": 150.0, "peak_speed_ms": 41.7, "avg_speed_kmh": 110.0,
        "peak_court": [5.4, 11.8], "peak_image": [820, 340], "duration_sec": 0.4,
    }
    vis = BallSpeedVisualizer(shots=[shot], frame_width=1920, frame_height=1080, language="en")

    def drew(frame_index):
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        vis.draw(frame, frame_index)
        return int(np.count_nonzero(frame != 40)) > 0

    assert drew(4) is True, "overlay should be active at output frame 4 (peak reached)"
    assert drew(3) is False, "overlay must NOT be active before the peak output frame"
    assert drew(6) is True, "overlay should still be visible within hold window after peak"
    print("  visualizer aligns to output frames: ok")


def main():
    tests = [
        test_constant_speed,
        test_two_shots_split_by_bounce,
        test_empty_and_degenerate,
        test_outlier_rejected,
        test_rally_gap_splits_rallies,
        test_output_frame_indices_track_compacted_position,
        test_visualizer_aligns_to_output_frames,
    ]
    for test in tests:
        print(f"-> {test.__name__}")
        test()
    print("\nAll ball speed tests passed.")


if __name__ == "__main__":
    main()
