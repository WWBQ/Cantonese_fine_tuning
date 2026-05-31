# 粤语微调 (Cantonese LLM Fine-tuning)

基于 Qwen2.5-7B-Instruct 微调的粤语翻译与对话模型。

## 项目简介

本项目使用 LoRA 对 Qwen2.5-7B-Instruct 进行微调，训练数据包含多个粤语-普通话平行语料库，总计约 213 万条训练样本。模型支持：

- 粤语 ↔ 普通话双向翻译
- 粤语对话
- 拒绝非法/危险请求

## 训练

- **基座模型**: Qwen2.5-7B-Instruct (4-bit 量化)
- **训练框架**: Unsloth + TRL (SFTTrainer)
- **微调方式**: LoRA (r=16, alpha=32)
- **训练步数**: 10,000 steps
- **硬件**: NVIDIA RTX 3090 (24GB)
- **数据量**: 213 万条

见 [`code/yue_train.ipynb`](code/yue_train.ipynb)

## 数据管线

1. 多源数据下载 → 2. 双向翻译对构造 (粤↔普 3:1) → 3. MD5 去重 → 4. 训练/验证/测试集划分

见 [`code/clean_yueyu.py`](code/clean_yueyu.py)

## 模型下载

模型文件托管在 Hugging Face：

- [模型链接] (待上传)

## 本地推理

```bash
# Ollama 部署
ollama create yue_qwen -f Modelfile
ollama run yue_qwen
```
