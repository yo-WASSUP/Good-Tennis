"""Summarize frame processing latency against the input video's deadline."""


def summarize_frame_times(frame_times, source_fps):
    ordered = sorted(frame_times)
    count = len(ordered)
    elapsed = sum(ordered)
    budget = 1.0 / source_fps

    def percentile(fraction):
        if not count:
            return None
        index = (count - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, count - 1)
        return (ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)) * 1000

    return {
        "frames": count,
        "source_fps": source_fps,
        "frame_budget_ms": budget * 1000,
        "frame_loop_seconds": elapsed,
        "frame_loop_fps": count / elapsed if elapsed else None,
        "latency_ms": {
            "mean": elapsed * 1000 / count if count else None,
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": ordered[-1] * 1000 if count else None,
        },
        "frames_over_budget": sum(seconds > budget for seconds in ordered),
        "timing_scope": "read/decode, detection/tracking, drawing, display (if enabled), temporary video write",
    }
