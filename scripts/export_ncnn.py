#!/usr/bin/env python3
"""Export all autoclip YOLO classify models to ncnn (.param/.bin/metadata.yaml).

Run inside the huji-algorithm venv:
    .venv/Scripts/python.exe scripts/export_ncnn.py [--out <dir>]

For each <sport>/<match_type> under src/resources/models/, exports best.pt to
ncnn format and normalizes the output layout to:
    <out>/<sport>/<match_type>/model.ncnn.param
    <out>/<sport>/<match_type>/model.ncnn.bin
    <out>/<sport>/<match_type>/metadata.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


MODELS_ROOT = Path(__file__).resolve().parent.parent / "src" / "resources" / "models"
MODEL_COMBOS = [
    ("ping_pong", "normal"),
    ("ping_pong", "profession"),
    ("badminton", "singles"),
    ("badminton", "doubles"),
]


def export_one(pt_path: Path, out_dir: Path) -> None:
    model = YOLO(str(pt_path))
    exported = Path(model.export(format="ncnn", imgsz=640))
    # ultralytics emits <stem>_ncnn_model/ with model.ncnn.{param,bin} + metadata.yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
        src = exported / name
        if not src.exists():
            raise FileNotFoundError(f"{name} missing in {exported}")
        shutil.copy2(src, out_dir / name)
    print(f"[ok] {pt_path.parent.relative_to(MODELS_ROOT)} -> {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_ROOT.parent / "resources" / "ncnn_models",
        help="output root (default: src/resources/ncnn_models)",
    )
    args = parser.parse_args()

    for sport, match_type in MODEL_COMBOS:
        pt_path = MODELS_ROOT / sport / match_type / "best.pt"
        if not pt_path.exists():
            print(f"[skip] missing {pt_path}", file=sys.stderr)
            continue
        export_one(pt_path, args.out / sport / match_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
