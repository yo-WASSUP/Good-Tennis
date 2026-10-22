"""Device selection shared by the YOLO detectors."""

from pathlib import Path


def is_coreml_model(model_path):
    return isinstance(model_path, (str, Path)) and Path(model_path).suffix.lower() in (".mlpackage", ".mlmodel")


def select_yolo_device(device="auto", model_path=None):
    """Use CPU host tensors for CoreML; otherwise prefer CUDA, MPS, CPU."""
    if device in (None, "auto"):
        device = "cpu"
        # CoreML owns accelerator scheduling. Moving host tensors through MPS
        # adds transfers before CoreML's NumPy/PIL input conversion.
        if is_coreml_model(model_path):
            return device, False
        try:
            import torch

            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except (ImportError, RuntimeError, AttributeError):
            device = "cpu"

    # Keep MPS in FP32, as on CPU. Existing CUDA inference retains FP16.
    half = str(device).split(":")[0] not in ("cpu", "mps")
    return device, half
