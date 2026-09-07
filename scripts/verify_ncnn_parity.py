#!/usr/bin/env python3
"""Parity check: ncnn vs ONNX Runtime on identical seeded letterbox input.

Run twice (separate processes — loading ORT and ncnn in one process is
unstable on Windows):
    .venv/Scripts/python.exe scripts/verify_ncnn_parity.py ort  <combo>...
    .venv/Scripts/python.exe scripts/verify_ncnn_parity.py ncnn <combo>...

Each run writes <combo>.npy logs to a temp dir; run both then call with
"compare" to assert argmax equality and max|Δlogit| < 1e-4.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

MODELS_ROOT = Path(__file__).resolve().parent.parent / "src" / "resources"
ONNX_ROOT = MODELS_ROOT / "models"
NCNN_ROOT = MODELS_ROOT / "ncnn_models"
COMBOS = [
    ("ping_pong", "normal"),
    ("ping_pong", "profession"),
    ("badminton", "singles"),
    ("badminton", "doubles"),
]
W = H = 640
TOL = 1e-4


def seeded_input() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def out_path(sport: str, match: str) -> Path:
    d = Path(tempfile.gettempdir()) / "huji_parity"
    d.mkdir(exist_ok=True)
    return d / f"{sport}_{match}.npy"


def run_ort(combo) -> np.ndarray:
    import onnxruntime as ort
    sport, match = combo
    sess = ort.InferenceSession(
        str(ONNX_ROOT / sport / match / "best.onnx"),
        providers=["CPUExecutionProvider"],
    )
    x = seeded_input()
    i, o = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    return sess.run([o], {i: x})[0][0]


def run_ncnn(combo) -> np.ndarray:
    import ncnn
    sport, match = combo
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # CPU mode for strict parity
    net.load_param(str(NCNN_ROOT / sport / match / "model.ncnn.param"))
    net.load_model(str(NCNN_ROOT / sport / match / "model.ncnn.bin"))
    x = seeded_input()
    mat = ncnn.Mat(np.ascontiguousarray(x[0]))
    ex = net.create_extractor()
    ex.input("in0", mat)
    _, out = ex.extract("out0")
    return np.array(out)


def main() -> int:
    mode = sys.argv[1]
    if mode not in ("ort", "ncnn", "compare"):
        print(__doc__)
        return 2

    if mode != "compare":
        for combo in COMBOS:
            out = run_ort(combo) if mode == "ort" else run_ncnn(combo)
            np.save(out_path(*combo), out)
            print(f"[{mode}] {combo[0]}/{combo[1]}: {np.round(out, 4).tolist()}")
        return 0

    failures = 0
    for sport, match in COMBOS:
        ort_out = np.load(out_path(sport, match))
        ncnn_out = np.load(out_path(sport, match))
        diff = float(np.max(np.abs(ort_out - ncnn_out)))
        same_argmax = int(np.argmax(ort_out)) == int(np.argmax(ncnn_out))
        ok = same_argmax and diff < TOL
        print(
            f"{'PASS' if ok else 'FAIL'} {sport}/{match}: "
            f"max|Δ|={diff:.2e} argmax={int(np.argmax(ort_out))}/"
            f"{int(np.argmax(ncnn_out))}"
        )
        if not ok:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
