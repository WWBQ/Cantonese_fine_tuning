#!/usr/bin/env python3
"""把 merged_model 转成 FP16 GGUF，再量化为 Q4_K_M。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "model" / "merged_model"
CONVERT = ROOT / "llama.cpp" / "convert_hf_to_gguf.py"
QUANT_BIN = ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize"
LIB_DIR = ROOT / "llama.cpp" / "build" / "bin"

FP16 = ROOT / "model" / "yue_qwen_fp16.gguf"
Q4 = ROOT / "model" / "yue_qwen_q4.gguf"


def main() -> None:
    if not MERGED.exists() or not any(MERGED.glob("*.safetensors")):
        raise SystemExit(f"找不到合并模型: {MERGED}")
    if not CONVERT.exists():
        raise SystemExit(f"找不到 {CONVERT}")
    if not QUANT_BIN.exists():
        raise SystemExit(f"找不到 {QUANT_BIN}")

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(LIB_DIR) + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")

    print(f"→ FP16 GGUF: {FP16}")
    subprocess.run(
        ["python3", str(CONVERT), str(MERGED), "--outfile", str(FP16), "--outtype", "f16"],
        check=True,
        env=env,
    )

    print(f"→ Q4_K_M: {Q4}")
    subprocess.run(
        [str(QUANT_BIN), str(FP16), str(Q4), "Q4_K_M"],
        check=True,
        env=env,
        cwd=str(LIB_DIR),
    )
    print(f"完成:\n  {FP16}\n  {Q4}")


if __name__ == "__main__":
    main()
