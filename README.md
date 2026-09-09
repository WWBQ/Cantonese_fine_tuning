# 粤语微调

基于 **Qwen2.5-7B-Instruct** 的 LoRA 微调，做成粤语助手（问答 / 写作 / 对话），而不是翻译机。

上一版用约 **213 万条粤普平行翻译** 训练，模型会把普通提问也拿去翻译。本轮改用约 **4.5 万条** 问答语料，并混入 **8,000** 条翻译以免完全忘掉翻译。行为评测：提问用粤语答、明确要求才翻译、违法请求拒绝。

权重体积太大，**不进 Git**。本仓库是数据清洗、训练脚本和实验记录。

## 训练配置

| 项 | 值 |
|---|---|
| 底模 | Qwen2.5-7B-Instruct（训练时 4-bit QLoRA） |
| 适配器 | LoRA r=16, α=32 |
| 数据 | `mix_train.jsonl` 51,701 条（问答 43,701 + 翻译 8,000） |
| 验证 | 问答 val 与翻译 val 分开记 loss |
| 合并 | 未量化 FP16 底模 + LoRA（不要在 4-bit 上 merge） |
| 导出 | `yue_qwen_fp16.gguf`、`yue_qwen_q4.gguf`（Q4_K_M） |

## 目录

```
notebooks/yue_train.ipynb   # 实际训练
scripts/prepare_qa.py       # 问答清洗 + 划分 + 混翻译
scripts/test_lora.py        # GPU 上测 LoRA
scripts/merge_model.py      # 合并到 FP16
scripts/to_gguf.py          # HF → GGUF → Q4_K_M
scripts/legacy/             # 旧翻译管线（不要再当主数据）
docs/DATA_CARD.md
deploy/Modelfile            # Ollama 配方（需本地 GGUF）
```

## 数据

```bash
python3 scripts/prepare_qa.py
```

产物在 `cleaned_data/qa/`（大体量 jsonl 不进 Git）。计数见 [`docs/DATA_CARD.md`](docs/DATA_CARD.md) 与 `cleaned_data/qa/stats.json`。

## 基座 vs 微调对比

```bash
python3 scripts/sample_compare_set.py          # 已生成 eval/compare_set.jsonl，可重抽
ollama pull qwen2.5:7b                         # 基座，与微调用同一 system
python3 scripts/compare_models.py --backend ollama \
  --base qwen2.5:7b --old yue-translator --ft yue_qwen
```

报告在 `eval/compare_report.md`。看「未要求翻译却短译」的条数，并人工扫探针题。


把 `yue_qwen_q4.gguf` 放到 `model/` 后：

```bash
cd model
ollama create yue_qwen -f ../deploy/Modelfile
ollama run yue_qwen
```

## 说明

- 微调改的是**行为**（粤语助手、别乱翻译），不补百科知识。领域知识应接 RAG。
- 不宣称母语级；语料来自已下载的开源粤语指令 / 对话 / 问答。
