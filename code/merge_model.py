import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# ---------- 设置镜像 ----------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ---------- 正确配置 ----------
# 关键修改：使用标准非量化版 Qwen2.5-7B-Instruct
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 本地 LoRA 适配器路径
LORA_PATH = "../model/my_model/outputs_yue_qwen/checkpoint-10000"
# 合并后模型保存路径
MERGED_PATH = "../model/merged_model"


# ----------------------------

def merge_qwen_lora_clean(lora_path, merged_path):
    print(f"📥 正在加载基座模型: {BASE_MODEL_NAME}")
    print(f"🔌 LoRA 适配器路径: {lora_path}")
    print(f"💾 合并输出路径: {merged_path}")

    # 1. 加载分词器
    print("\n📝 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    # 2. 加载干净的基座模型 (FP16)
    print("\n🚀 正在下载并加载基座模型 (FP16 模式)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16,  # <-- torch_dtype 改为 dtype
        device_map="cpu",  # <-- 直接用字符串 "cpu"
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. 挂载 LoRA 适配器
    print("\n🧩 正在挂载并合并 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()

    # 4. 保存最终模型
    print(f"\n💾 正在保存合并模型到: {merged_path}")
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_path)

    print("\n🎉 合并成功！")
    print(f"📁 模型路径: {merged_path}")


if __name__ == "__main__":
    # 如果下载慢，取消下面这行的注释设置镜像
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    merge_qwen_lora_clean(LORA_PATH, MERGED_PATH)