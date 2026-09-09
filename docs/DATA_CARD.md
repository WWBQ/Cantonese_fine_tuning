# Data Card：粤语问答 SFT

> 生成脚本：`python3 scripts/prepare_qa.py`（seed=42）。大体量 jsonl 不进 Git，计数以 `cleaned_data/qa/stats.json` 为准。

## 这批数据是什么

给 **问答 / 写作 / 对话** 用的监督微调样本，不是粤普平行翻译。上一版 `cleaned_data/yue_train.jsonl`（约 213 万条）几乎全是「请翻译」，训完模型只会翻译。本卡对应产物 `cleaned_data/qa/`。本轮已按 `mix_train.jsonl` 训完 LoRA 并导出 GGUF。

公开边界：语料来自已下载的开源粤语指令/对话/百科问答，**不是客户数据**。粤语真实性靠汉字过滤，没有人工逐条核验，不夸「母语级」。

## 来源与保留数

| 来源 | 原始 | 保留 | 任务标签 | 说明 |
|---|---:|---:|---|---|
| yue-alpaca | 18,649 | 17,544 | instruct | 广告、建议、写作；大量过短 output 被丢 |
| Cantonese-Dialogue | 15,087 | 15,009 | dialogue | 粤语对话问答 |
| AllAspectQA 11K | 10,998 | 10,991 | qa | 去掉表头一行 |
| cantonesewiki_doyouknow | 1,569 | 1,568 | qa | 用 FormattedResponse |
| yue_harmful_behaviors | 416 | 405 | refuse | 统一拒答，不套「请翻译」指令 |
| **合计** | **46,719** | **45,517** | | 通过率 **97.4%** |

未纳入：`english_cantonese_translation.csv`、Wiki 平行句、raptorkwok / agentlans / botisan（那些是翻译，仍在 `data/` 与 `cleaned_data/`）。

## 清洗规则（阈值有记录）

1. 空白折叠；`<noinput>` 视为空 input  
2. 丢掉表头（Question/Response）  
3. instruction 长度 4–2000；output 长度 8–8000  
4. instruction 与 output 完全相同 → 丢  
5. 指令和回复都没有汉字 → 丢  
6. `sha1(instruction+input+output)` 去重  

主要刷掉的是 alpaca 里 **1,082** 条过短回复（`out_short`），其余各源丢弃都很少。

## 划分

按 **source** 分层，`random.seed(42)`，约 96% / 2% / 2%：

| split | 条数 |
|---|---:|
| train | 43,701 |
| val | 908 |
| test | 908 |

另：`mix_train.jsonl` = train + 从 `cleaned_data/yue_train.jsonl` **水库抽样 8,000** 条翻译（约 15.5%），避免下一轮完全忘掉翻译。抽译样本的 system 也改成粤语助手，不再写「翻译助手」。

## 样本格式

```json
{
  "id": "sha1 前 16 位",
  "instruction": "用户问题或任务",
  "input": "可选上下文，可空",
  "output": "粤语回复",
  "source": "yue-alpaca",
  "task": "instruct",
  "text": "<|im_start|>system\\n...ChatML..."
}
```

系统句：

> 你是一个粤语助手。用粤语回答问题、完成写作或对话。只有用户明确要求翻译时才翻译。不要把普通提问当成翻译任务。

`text` 里 **user 段包含 instruction**（上一版笔记本只喂了 input，等于没告诉模型要做什么）。

## 限制

- AllAspectQA / doyouknow 有「嘩好耐冇见」套话，偏闲聊，不是金融事实库  
- 拒答只有 405 条，覆盖很窄  
- 没有回译一致性、没有 MinHash 近重（本轮只用精确 hash）  
- 拒答覆盖仍窄；推理侧可用 Ollama system 再收紧
- 知识缺口请接 RAG，不要指望再堆翻译语料
