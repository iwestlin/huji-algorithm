import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.main.constant.autoclip_constant import (
    ActionType,
    badminton_classes_mapping,
    ping_pong_classes_mapping,
)
from src.main.service.large_model_service import NcnnModelPredictor, letterbox_image
from src.test.base.test_base import TestCaseBase


RESOURCES = Path(__file__).resolve().parents[2] / "resources"
NCNN_ROOT = RESOURCES / "ncnn_models"


def _make_test_image() -> str:
    """随机 RGB 640x480 图,写为 png。"""
    img = np.random.default_rng(42).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    f = Path(tempfile.gettempdir()) / "huji_ncnn_pred_test.png"
    cv2.imwrite(str(f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(f)


class TestLetterbox(unittest.TestCase):
    def test_letterbox_shape_and_pad(self):
        arr = letterbox_image(_make_test_image(), size=640)
        self.assertEqual(arr.shape, (3, 640, 640))
        self.assertEqual(arr.dtype, np.float32)
        # 640x480 → scale=1.0? 不:短边 480 → scale=640/480=1.333,newH=640,newW=853>640
        # → scale 收敛为 640/640=1.0;宽图:上下 pad 应为 114/255
        self.assertAlmostEqual(float(arr[:, : (640 - 480) // 2, :].mean()), 114 / 255, places=2)


class TestNcnnPredictor(TestCaseBase):
    def _setup_internal(self):
        pass

    def test_predict_returns_known_action(self):
        param = str(NCNN_ROOT / "ping_pong" / "normal" / "model.ncnn.param")
        predictor = NcnnModelPredictor(param, debug=False)
        action = predictor.predict(_make_test_image(), ping_pong_classes_mapping)
        self.assertIsInstance(action, ActionType)

    def test_predict_badminton(self):
        param = str(NCNN_ROOT / "badminton" / "singles" / "model.ncnn.param")
        predictor = NcnnModelPredictor(param, debug=False)
        action = predictor.predict(_make_test_image(), badminton_classes_mapping)
        self.assertIsInstance(action, ActionType)
