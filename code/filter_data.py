import json
import hashlib

file_path = '../cleaned_data/final_train_data.jsonl'
output_file_path = '../cleaned_data/final_data.jsonl'

# 1. 改用 set，查询效率提升万倍
id_set = set()
count_total = 0
count_kept = 0

with open(output_file_path, 'w', encoding='utf-8') as wf:
    # 2. 改用文件迭代器，防止 readlines() 一次性撑爆内存
    with open(file_path, 'r', encoding='utf-8') as rf:
        for line in rf:
            count_total += 1
            try:
                data = json.loads(line)

                # 检查字段是否存在
                if not data.get('input') or not data.get('output'):
                    continue

                # 3. 生成 MD5
                content = data['input'] + data['output']
                id_ = hashlib.md5(content.encode('utf-8')).hexdigest()

                # 4. O(1) 极速查询
                if id_ in id_set:
                    continue

                id_set.add(id_)
                wf.write(line)  # line 自带换行符
                count_kept += 1

                # 每 10 万条打印一次进度
                if count_kept % 100000 == 0:
                    print(f"已处理 {count_total} 条，保留 {count_kept} 条...")

            except json.JSONDecodeError:
                continue

print(f"✅ 去重完成！原始: {count_total} -> 保留: {count_kept} (删除了 {count_total - count_kept} 条重复项)")