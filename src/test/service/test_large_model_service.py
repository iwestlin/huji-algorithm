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

    def test_preprocess_16x9_is_center_crop_not_letterbox(self):
        """YOLO classify 是短边缩放到 size 后中心裁剪,不能 letterbox pad 114。

        16:9 画布:左红右蓝、上下绿。center-crop 丢掉左右色条;
        letterbox 会保留完整画面并在上下填 114。
        """
        h, w, size = 180, 320, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (0, 255, 0)
        img[:, :20] = (255, 0, 0)
        img[:, -20:] = (0, 0, 255)
        f = Path(tempfile.gettempdir()) / "huji_ncnn_pred_test_16x9.png"
        cv2.imwrite(str(f), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        arr = preprocess_image(str(f), size=size)
        self.assertEqual(arr.shape, (3, size, size))
        rgb = (arr.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        # 中心应是绿色场地,不是 letterbox 的 114 灰边
        center = rgb[size // 2, size // 2]
        self.assertGreater(int(center[1]), 200)
        self.assertLess(int(center[0]), 40)
        # 左右边缘不应残留红/蓝边条(那是 letterbox 才会留下的完整画面)
        left_col = rgb[:, 0]
        right_col = rgb[:, -1]
        self.assertFalse(np.all(left_col[:, 0] > 200), "left edge is red — letterbox, not crop")
        self.assertFalse(np.all(right_col[:, 2] > 200), "right edge is blue — letterbox, not crop")
        # 上下边缘也不应是 pad=114 的灰带
        top_mean = float(rgb[0, :, :].mean())
        self.assertGreater(abs(top_mean - 114.0), 20.0, "top row looks like letterbox pad 114")


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
