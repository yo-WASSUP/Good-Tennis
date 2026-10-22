import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

from tennis_analysis.detection.device import select_yolo_device


class DeviceSelectionTests(unittest.TestCase):
    def torch_backend(self, cuda=False, mps=False):
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=cuda)),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps))),
        )

    def test_apple_gpu_uses_fp32(self):
        with patch.dict(sys.modules, torch=self.torch_backend(mps=True)):
            self.assertEqual(select_yolo_device(), ("mps", False))

    def test_cuda_keeps_priority_and_fp16(self):
        with patch.dict(sys.modules, torch=self.torch_backend(cuda=True, mps=True)):
            self.assertEqual(select_yolo_device(), (0, True))

    def test_no_accelerator_uses_cpu(self):
        with patch.dict(sys.modules, torch=self.torch_backend()):
            self.assertEqual(select_yolo_device(None), ("cpu", False))

    def test_older_torch_without_mps_uses_cpu(self):
        backend = self.torch_backend()
        backend.backends = SimpleNamespace()
        with patch.dict(sys.modules, torch=backend):
            self.assertEqual(select_yolo_device(), ("cpu", False))

    def test_missing_torch_uses_cpu(self):
        with patch.dict(sys.modules, torch=None):
            self.assertEqual(select_yolo_device(), ("cpu", False))

    def test_explicit_device_is_preserved_without_probing(self):
        backend = self.torch_backend(cuda=True, mps=True)
        with patch.dict(sys.modules, torch=backend):
            for device, half in (("cpu", False), ("mps", False), ("mps:0", False), (0, True), ("cuda:1", True)):
                with self.subTest(device=device):
                    self.assertEqual(select_yolo_device(device), (device, half))
        backend.cuda.is_available.assert_not_called()
        backend.backends.mps.is_available.assert_not_called()

    def test_coreml_keeps_host_tensors_on_cpu_even_when_gpu_available(self):
        backend = self.torch_backend(cuda=True, mps=True)
        with patch.dict(sys.modules, torch=backend):
            for model in ("weights/model.mlpackage", Path("weights/model.mlmodel")):
                with self.subTest(model=model):
                    self.assertEqual(select_yolo_device(model_path=model), ("cpu", False))
        backend.cuda.is_available.assert_not_called()
        backend.backends.mps.is_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
