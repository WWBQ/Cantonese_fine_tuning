import json
import os
from pathlib import Path

import torch
from transformers import logging
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from safetensors.torch import load_file

# 关闭所有警告
logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "model" / "merged_model")

print("🔧 正在加载模型（仅需一次，持续对话）...")

# 加载配置
with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
    config_dict = json.load(f)
config = Qwen2Config(**config_dict)
model = Qwen2ForCausalLM(config).to(dtype=torch.float16)

# 加载权重
safetensors_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
state_dict = {}
for f in safetensors_files:
    state_dict.update(load_file(os.path.join(MODEL_PATH, f)))
model.load_state_dict(state_dict, strict=True)
model.eval()

# 加载 Tokenizer
with open(os.path.join(MODEL_PATH, "tokenizer_config.json"), "r") as f:
    tokenizer_config = json.load(f)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(MODEL_PATH, "tokenizer.json"),
    **tokenizer_config
)

# 设置 pad_token，消除警告
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ 模型就绪！输入 'exit' 退出\n")

# 交互循环
system_prompt = (
    "你是一个粤语助手。用粤语回答问题、完成写作或对话。"
    "只有用户明确要求翻译时才翻译。不要把普通提问当成翻译任务。"
)

while True:
    user_input = input("👤 你: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 再见！")
        break

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    # 添加生成提示
    print("🤖 粤语助手: ", end="", flush=True)

    # 使用 Streamer 实现逐字输出
    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer
    )
    print()