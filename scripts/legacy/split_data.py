import json
import random


def professional_split(input_file, train_ratio=0.98, val_ratio=0.01):
    with open(input_file, 'r', encoding='utf-8') as f:
        all_data = f.readlines()

    # 彻底打乱数据，确保每个集里都有各种来源的语料
    random.seed(42)  # 固定随机种子，方便以后复现
    random.shuffle(all_data)

    total = len(all_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    # 保存三个文件
    files = {
        '../cleaned_data/yue_train.jsonl': train_data,
        '../cleaned_data/yue_val.jsonl': val_data,
        '../cleaned_data/yue_test.jsonl': test_data
    }

    for path, data in files.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(data)
        print(f"✅ 已生成 {path}，样本数: {len(data)}")


professional_split('../cleaned_data/final_data.jsonl')