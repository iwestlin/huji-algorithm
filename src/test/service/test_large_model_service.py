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
from src.main.service.large_model_service import NcnnModelPredictor, preprocess_image
from src.test.base.test_base import TestCaseBase


RESOURCES = Path(__file__).resolve().parents[2] / "resources"
NCNN_ROOT = RESOURCES / "ncnn_models"


def _make_test_image() -> str:
    """随机 RGB 640x480 图,写为 png。"""
    img = np.random.default_rng(42).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    f = Path(tempfile.gettempdir()) / "huji_ncnn_pred_test.png"
    cv2.imwrite(str(f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(f)


class TestPreprocess(unittest.TestCase):
    def test_preprocess_shape_and_center_crop(self):
        # 640x480 → 短边 480 → scale=640/480 → 853x640 → CenterCrop(640) 居中取 640x640
        arr = preprocess_image(_make_test_image(), size=640)
        self.assertEqual(arr.shape, (3, 640, 640))
        self.assertEqual(arr.dtype, np.float32)

    def test_preprocess_square_passthrough(self):
        # 正方形输入:Resize 后即 640x640,CenterCrop 无裁切
        img = np.random.default_rng(1).integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
        f = Path(tempfile.gettempdir()) / "huji_ncnn_pred_test_sq.png"
        cv2.imwrite(str(f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        arr = preprocess_image(str(f), size=640)
        self.assertEqual(arr.shape, (3, 640, 640))


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
