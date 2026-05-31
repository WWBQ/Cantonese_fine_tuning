# convert_to_gguf.py
import subprocess
import os

MODEL_PATH = "/Users/chengfeng/Desktop/粤语微调/model/merged_model"
OUTPUT_PATH = "/Users/chengfeng/Desktop/粤语微调/model/yue_qwen_q4.gguf"

# 先用 transformers 以标准格式重新保存一次
# 然后调用 llama.cpp 的 convert_hf_to_gguf.py
subprocess.run([
    "python3",
    "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/llama_cpp/convert.py",
    MODEL_PATH,
    "--outfile", OUTPUT_PATH,
    "--outtype", "q4_k_m"
])
print(f"✅ 转换完成: {OUTPUT_PATH}")

