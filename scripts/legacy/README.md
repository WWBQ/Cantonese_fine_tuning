# 旧翻译管线

`clean_yueyu.py` → `filter_data.py` → `split_data.py` 产出 `cleaned_data/yue_*.jsonl`（约 213 万条粤普翻译）。

`train_yue.py` 是 Llama-3 模板草稿，真正训练用的是 `notebooks/yue_train.ipynb`（Qwen2.5）。

新训练请用 `scripts/prepare_qa.py` 的产物，不要再拿这份翻译集当主数据。
