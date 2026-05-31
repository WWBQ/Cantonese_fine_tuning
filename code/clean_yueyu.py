import json
import random
import os
from opencc import OpenCC
import pandas as pd
# ================= 配置区 =================
DATA_DIR = "/Users/chengfeng/Desktop/粤语微调/data"
OUTPUT_FILE = "../cleaned_data/final_train_data.jsonl"

# 设定比例：每 3 条粤转普，配 1 条普转粤 (3:1)
REVERSE_RATIO = 0.33
t2s = OpenCC('t2s')

# ==========================================

def get_instruction(task_type):
    """根据任务类型随机选择指令，增加模型泛化能力"""
    if task_type == "yue_to_zh":
        return random.choice([
            "请将这段地道的粤语翻译成普通话。",
            "帮我把这段话翻译成普通话。",
            "用普通话怎么说这段粤语？",
            "翻译以下粤语内容："
        ])
    elif task_type == "zh_to_yue":
        return random.choice([
            "请将这段普通话翻译成地道的粤语。",
            "帮我把这段话转成粤语口语。",
            "用粤语怎么说这段话？",
            "将以下内容翻译为粤语："
        ])
    return "请根据提示进行操作。"


def process_item(yue, zh):
    """构造双向数据"""
    items = []
    # 核心任务：粤转普 (必须包含)
    zh_s = t2s.convert(zh.strip())
    items.append({
        "instruction": get_instruction("yue_to_zh"),
        "input": yue.strip(),
        "output": zh_s.strip()
    })

    # 镜像任务：普转粤 (按概率构造)
    if random.random() < REVERSE_RATIO:
        items.append({
            "instruction": get_instruction("zh_to_yue"),
            "input": zh_s.strip(),
            "output": yue.strip()
        })
    return items


def main():
    final_dataset = []

    # 1. 处理 botisan-ai 格式 (嵌套 translation 字典)
    botisan_file = os.path.join(DATA_DIR, "botisan-ai:cantonese-mandarin-translations.json")
    with open(botisan_file, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line.strip())
            data = raw.get("translation", {})
            final_dataset.extend(process_item(data["yue"], data["zh"]))
    print(f"✅ 已加载 Botisan 数据，当前总数: {len(final_dataset)}")



    data_wiki = os.path.join(DATA_DIR, "data-wiki.json")
    with open(data_wiki, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            final_dataset.extend(process_item(data["yue"], data["zh"]))

    print(f"✅ 已加载 data-wiki 数据，当前总数: {len(final_dataset)}")


    parallel_corpus = [
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus/test.json"),
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus/train.json"),
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus/validation.json")
    ]
    for corpus in parallel_corpus:
        with open(corpus, "r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line.strip())
                data = raw.get("translation", {})
                final_dataset.extend(process_item(data["yue"], data["zh"]))
        print(f"✅ 已加载 parallel-corpus {corpus} 数据，当前总数: {len(final_dataset)}")


    agentlans = [
        os.path.join(DATA_DIR, "agentlans:cantonese-chinese/test.jsonl.gz"),
        os.path.join(DATA_DIR, "agentlans:cantonese-chinese/train.jsonl.gz"),
        os.path.join(DATA_DIR, "agentlans:cantonese-chinese/validation.jsonl.gz")
    ]
    for corpus1 in agentlans:
        df = pd.read_json(corpus1, lines=True, compression='gzip')
        for index, row in df.iterrows():
            yue = row['yue']
            zh = row['zht']
            final_dataset.extend(process_item(yue, zh))
        print(f"✅ 已加载 agentlans {corpus1} 数据，当前总数: {len(final_dataset)}")



    parallel_corpus_gen3s = [
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus-gen3/test-00000-of-00001.parquet"),
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus-gen3/train-00000-of-00001.parquet"),
        os.path.join(DATA_DIR, "raptorkwok:cantonese-traditional-chinese-parallel-corpus-gen3/validation-00000-of-00001.parquet")
    ]
    for corpus2 in parallel_corpus_gen3s:
        df = pd.read_parquet(corpus2, engine='pyarrow')
        for index, row in df.iterrows():
            yue = row['yue']
            zh = row['ref']
            final_dataset.extend(process_item(yue, zh))
        print(f"✅ 已加载 parallel_corpus_gen3s {corpus2} 数据，当前总数: {len(final_dataset)}")

    # -------------------------------------------------------

    # 3. 处理安全/拒绝类数据 (不进行镜像构造，只单向拒绝)
    safety_file = os.path.join(DATA_DIR, "yue_harmful_behaviors.csv")  # 请确认文件名
    df = pd.read_csv(safety_file)
    for _, row in df.iterrows():
        final_dataset.append({
            "instruction": get_instruction("yue_to_zh"),
            "input": row['text'],
            "output": "对不起，作为一个人工智能助手，我不能协助你进行此类违规或危险的行为。"
        })
        print(f"✅ 已加入安全对齐数据，当前总数: {len(final_dataset)}")

    # -------------------------------------------------------
    # 4. 彻底打乱数据 (防止模型学习到顺序偏置)
    print("正在进行最后的数据随机打乱...")
    random.shuffle(final_dataset)

    # 5. 导出为 JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"🚀 全量数据处理完成！")
    print(f"📂 文件已保存至: {OUTPUT_FILE}")
    print(f"📊 样本总计: {len(final_dataset)} 条")


if __name__ == "__main__":
    main()