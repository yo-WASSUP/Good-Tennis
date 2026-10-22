import unittest

from tennis_analysis.analysis.performance import summarize_frame_times


class FrameTimingTests(unittest.TestCase):
    def test_fast_average_does_not_hide_missed_deadlines(self):
        result = summarize_frame_times([0.01, 0.01, 0.01, 0.09], 25)
        self.assertAlmostEqual(result["frame_loop_fps"], 100 / 3)
        self.assertEqual(result["frames_over_budget"], 1)
        self.assertAlmostEqual(result["latency_ms"]["p95"], 78)
        self.assertEqual(result["frame_budget_ms"], 40)

    def test_empty_video_has_no_latency_or_fps(self):
        result = summarize_frame_times([], 25)
        self.assertEqual(result["frames"], 0)
        self.assertIsNone(result["frame_loop_fps"])
        self.assertIsNone(result["latency_ms"]["p95"])

    def test_single_frame_at_deadline(self):
        result = summarize_frame_times([0.04], 25)
        self.assertEqual(result["frame_loop_fps"], 25)
        self.assertEqual(result["frames_over_budget"], 0)
        self.assertEqual(result["latency_ms"]["p95"], 40)


if __name__ == "__main__":
    unittest.main()
