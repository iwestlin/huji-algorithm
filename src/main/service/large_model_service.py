# ncnn python binding 无类型桩,strict 下豁免对 ncnn 的访问检查
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
from pathlib import Path
from typing import Any

import ncnn
import numpy as np
import ruamel.yaml
from PIL import Image

from src.main.config.config import LargeModelServiceConfig, ModelConfig
from src.main.constant.autoclip_constant import ActionType
from src.main.logger import LOG


INPUT_SIZE = 640


def preprocess_image(
    img_path: str, size: int = INPUT_SIZE
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """复刻 ultralytics classify 预测预处理(模型 .pt 内嵌 transforms)。

    Resize(短边=size,长边 int 截断,PIL BILINEAR) → CenterCrop(size) → /255 → CHW。
    与 torchvision Resize+CenterCrop+ToTensor 逐位一致(经逐帧 parity 验证)。
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    x_off, y_off = (new_w - size) // 2, (new_h - size) // 2
    cropped = resized.crop((x_off, y_off, x_off + size, y_off + size))
    return (np.asarray(cropped, dtype=np.float32) / 255.0).transpose(2, 0, 1)


class NcnnModelPredictor:
    """ncnn 分类推理(替代 ultralytics YOLO),行为镜像 app 侧 Dart 实现。"""

    def __init__(self, param_path: str, debug: bool, confidence_threshold: float = 0.0) -> None:
        self.debug = debug
        self.confidence_threshold = confidence_threshold
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
            # ProcessPool 已经按片段并行,每个 net 只用 1 线程避免 18 进程打满全部核
            net.opt.num_threads = 1
            net.load_param(self.param_path)
            net.load_model(self.bin_path)
            self._net = net
        return self._net

    def predict(self, img_path: str, classes_mapping: dict[str, ActionType]) -> ActionType:
        x = preprocess_image(img_path)
        # ncnn.Mat 包装而非拷贝 numpy 缓冲区:input_arr 必须存活到 extract 结束,
        # 否则临时数组被 GC 后读到已释放内存(表现为不确定的 NaN/错分类)
        input_arr = np.ascontiguousarray(x)
        mat = ncnn.Mat(input_arr)
        ex = self._get_net().create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        # ncnn 输出 Mat 为 1-D(类别数);reshape(-1) 同时兼容 (C,) 与 (1, C)
        logits = np.array(out).reshape(-1)
        # 模型导出时最后一层已带 Softmax;若发现不是概率分布则补做一次
        if float(logits.min()) >= -1e-6 and abs(float(logits.sum()) - 1.0) < 1e-3:
            probs = logits
        else:
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        top1_index = int(np.argmax(probs))
        top1 = self.names[top1_index]
        if self.debug:
            LOG.info(
                f"ncnn predict {img_path}: top1={top1}"
                f" prob={probs[top1_index]:.4f} logits={np.round(logits, 4).tolist()}"
            )
        if top1 in classes_mapping:
            action = classes_mapping[top1]
            # 低置信度的 play_ball 多为死球/等待画面,置为 UNKNOWN 以结束回合,
            # 避免死球后仍被判定为比赛而保留过长片段。
            if (
                action == ActionType.PLAY_BALL
                and float(probs[top1_index]) < self.confidence_threshold
            ):
                return ActionType.UNKNOWN
            return action
        raise ValueError(f"Unknown action type: {top1}")


class LargeModelService:
    def __init__(self, large_model_service_config: LargeModelServiceConfig) -> None:
        self.models: dict[str, ModelConfig] = large_model_service_config.models
        self.debug = large_model_service_config.debug
        self.confidence_threshold = large_model_service_config.confidence_threshold

    def get_predictor(self, model_name: str) -> NcnnModelPredictor:
        model_path = self.get_model_path(model_name)
        return NcnnModelPredictor(model_path, self.debug, self.confidence_threshold)

    def get_model_path(self, model_name: str) -> str:
        return self.models[model_name].predict_model_path
