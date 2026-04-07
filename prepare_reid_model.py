#!/usr/bin/env python3
"""One-shot: export OSNet-x0.25 → ONNX → instruct user to run trtexec.

Run once before using zed-color.py:
    python3 prepare_reid_model.py

Then run the printed trtexec command to build the TensorRT engine (~2 min on Orin).
"""
import os
import torch
import torchreid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(SCRIPT_DIR, "osnet_x025_reid.onnx")
ENGINE_PATH = os.path.join(SCRIPT_DIR, "osnet_x025_reid.engine")

print("Loading OSNet-x0.25 from torchreid (weights auto-downloaded if needed)...")
model = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=True,
)
model.eval()

print(f"Exporting to ONNX: {ONNX_PATH}")
dummy = torch.zeros(1, 3, 256, 128)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["embedding"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
)
print(f"ONNX saved: {ONNX_PATH}")
print()
print("Now run this to build the TensorRT engine (one-time, ~2 min on Orin):")
print()
print(
    f"  trtexec --onnx={ONNX_PATH} "
    f"--saveEngine={ENGINE_PATH} "
    "--fp16 --workspace=512"
)
print()
print(f"When done, {ENGINE_PATH} must exist before running zed-color.py.")
