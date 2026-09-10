# ncnn python binding 无类型桩,strict 下豁免对 ncnn 的访问检查
# (与 auto_clipper.py 中 ray 的逐行 ignore 同目的)
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
from pathlib import Path
from typing import Any

import cv2
import ncnn
import numpy as np
import ruamel.yaml

from src.main.config.config import LargeModelServiceConfig, ModelConfig
from src.main.constant.autoclip_constant import ActionType
from src.main.logger import LOG


INPUT_SIZE = 640
PAD_VALUE = 114


def letterbox_image(img_path: str, size: int = INPUT_SIZE) -> np.ndarray[Any, np.dtype[np.float32]]:
    """YOLO classify letterbox:短边缩放到 size,pad 114 居中,/255,CHW。

    与 huji-app 的 ImagePreprocessor 行为一致。
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError(f"Unable to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    new_w, new_h = max(1, min(new_w, size)), max(1, min(new_h, size))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)


class NcnnModelPredictor:
    """ncnn 分类推理(替代 ultralytics YOLO),行为镜像 app 侧 Dart 实现。"""

    def __init__(self, param_path: str, debug: bool) -> None:
        self.debug = debug
        self.param_path = param_path
        self.bin_path = str(Path(param_path).with_suffix(".bin"))
        metadata_path = Path(param_path).parent / "metadata.yaml"
        yaml = ruamel.yaml.YAML(typ="safe", pure=True)
        with open(metadata_path) as f:
            metadata: Any = yaml.load(f)
        self.names: dict[int, str] = {int(k): v for k, v in metadata["names"].items()}
        self._net: Any = None

    def _get_net(self) -> Any:
        if self._net is None:
            net = ncnn.Net()
            net.opt.use_vulkan_compute = False  # 与 parity 脚本一致,CPU 保证确定性
            net.load_param(self.param_path)
            net.load_model(self.bin_path)
            self._net = net
        return self._net

    def predict(self, img_path: str, classes_mapping: dict[str, ActionType]) -> ActionType:
        x = letterbox_image(img_path)
        mat = ncnn.Mat(np.ascontiguousarray(x))
        ex = self._get_net().create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        logits = np.array(out)[0]
        top1 = self.names[int(np.argmax(logits))]
        if self.debug:
            LOG.info(f"ncnn predict {img_path}: top1={top1} logits={np.round(logits, 4).tolist()}")
        if top1 in classes_mapping:
            return classes_mapping[top1]
        raise ValueError(f"Unknown action type: {top1}")


class LargeModelService:
    def __init__(self, large_model_service_config: LargeModelServiceConfig) -> None:
        self.models: dict[str, ModelConfig] = large_model_service_config.models
        self.debug = large_model_service_config.debug

    def get_predictor(self, model_name: str) -> NcnnModelPredictor:
        model_path = self.get_model_path(model_name)
        return NcnnModelPredictor(model_path, self.debug)

    def get_model_path(self, model_name: str) -> str:
        return self.models[model_name].predict_model_path
