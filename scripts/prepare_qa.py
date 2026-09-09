#!/usr/bin/env python3
"""把仓库里已有的粤语问答语料洗成统一 SFT 格式，并划分 train/val/test。

上一版训练只用了粤普平行翻译，问答数据在 data/ 里从未进训练集。
本脚本只处理问答 / 指令 / 对话 / 拒答，不把 200 万条翻译对混进来
（可用 --mix-translate N 从 cleaned_data/tr/yue_train.jsonl 抽 N 条，避免完全忘掉翻译）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"
OUT = ROOT / "cleaned_data" / "qa"

SYSTEM_PROMPT = (
    "你是一个粤语助手。用粤语回答问题、完成写作或对话。"
    "只有用户明确要求翻译时才翻译。不要把普通提问当成翻译任务。"
)

CJK_RE = re.compile(r"[\u3400-\u9fff]")
WS_RE = re.compile(r"\s+")
PLACEHOLDER_INPUT = {"", "<noinput>", "<no input>", "none", "null", "n/a"}

MIN_INST, MAX_INST = 4, 2000
MIN_OUT, MAX_OUT = 8, 8000

csv.field_size_limit(sys.maxsize)


def norm_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = WS_RE.sub(" ", text).strip()
    return text


def norm_input(value: object) -> str:
    text = norm_text(value)
    if text.lower() in PLACEHOLDER_INPUT:
        return ""
    return text


def record_id(instruction: str, inp: str, output: str) -> str:
    blob = f"{instruction}\n{inp}\n{output}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def to_chatml(instruction: str, inp: str, output: str) -> str:
    user = f"{instruction}\n\n{inp}" if inp else instruction
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def make_record(instruction: str, inp: str, output: str, source: str, task: str) -> dict:
    return {
        "id": record_id(instruction, inp, output),
        "instruction": instruction,
        "input": inp,
        "output": output,
        "source": source,
        "task": task,
        "text": to_chatml(instruction, inp, output),
    }


def drop_reason(instruction: str, inp: str, output: str) -> str | None:
    if instruction.lower() in {"question", "prompt", "instruction"}:
        return "header"
    if output.lower() in {"response", "output", "answer"}:
        return "header"
    if len(instruction) < MIN_INST:
        return "inst_short"
    if len(instruction) > MAX_INST:
        return "inst_long"
    if len(output) < MIN_OUT:
        return "out_short"
    if len(output) > MAX_OUT:
        return "out_long"
    if instruction == output:
        return "identity"
    if not CJK_RE.search(instruction) and not CJK_RE.search(output):
        return "no_cjk"
    return None


def accept(raw_inst, raw_inp, raw_out, source: str, task: str, seen: set[str], dropped: Counter) -> dict | None:
    dropped["raw"] += 1
    instruction = norm_text(raw_inst)
    inp = norm_input(raw_inp)
    output = norm_text(raw_out)
    reason = drop_reason(instruction, inp, output)
    if reason:
        dropped[reason] += 1
        return None
    rec = make_record(instruction, inp, output, source, task)
    if rec["id"] in seen:
        dropped["duplicate"] += 1
        return None
    seen.add(rec["id"])
    dropped["kept"] += 1
    return rec


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_alpaca(seen: set[str]) -> tuple[list[dict], Counter]:
    dropped = Counter()
    rows: list[dict] = []
    path = RAW / "yue-alpaca" / "train.jsonl"
    for item in iter_jsonl(path):
        rec = accept(
            item.get("instruction"),
            item.get("input"),
            item.get("output"),
            "yue-alpaca",
            "instruct",
            seen,
            dropped,
        )
        if rec:
            rows.append(rec)
    return rows, dropped


def load_dialogue(seen: set[str]) -> tuple[list[dict], Counter]:
    dropped = Counter()
    rows: list[dict] = []
    path = RAW / "Cantonese-Dialogue" / "train.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        rec = accept(
            item.get("instruction"),
            item.get("input"),
            item.get("output"),
            "cantonese-dialogue",
            "dialogue",
            seen,
            dropped,
        )
        if rec:
            rows.append(rec)
    return rows, dropped


def load_allaspect(seen: set[str]) -> tuple[list[dict], Counter]:
    dropped = Counter()
    rows: list[dict] = []
    path = RAW / "Cantonese_AllAspectQA_11K" / "cantonese_allaspectqa_11k.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        rec = accept(
            item.get("Question"),
            "",
            item.get("Response"),
            "allaspect-qa",
            "qa",
            seen,
            dropped,
        )
        if rec:
            rows.append(rec)
    return rows, dropped


def load_doyouknow(seen: set[str]) -> tuple[list[dict], Counter]:
    dropped = Counter()
    rows: list[dict] = []
    path = RAW / "cantonesewiki_doyouknow" / "cantonesewiki_doyouknow.csv"
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            rec = accept(
                item.get("Prompt"),
                "",
                item.get("FormattedResponse") or item.get("RawResponse"),
                "wiki-doyouknow",
                "qa",
                seen,
                dropped,
            )
            if rec:
                rows.append(rec)
    return rows, dropped


REFUSAL = (
    "唔好意思，呢类违法或者危险嘅请求我帮唔到。"
    "如果你系想倾合法、安全嘅问题，我可以用粤语同你倾。"
)


def load_refuse(seen: set[str]) -> tuple[list[dict], Counter]:
    dropped = Counter()
    rows: list[dict] = []
    path = RAW / "yue_harmful_behaviors.csv"
    if not path.exists():
        return rows, dropped
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        field = "text" if reader.fieldnames and "text" in reader.fieldnames else (reader.fieldnames or [None])[0]
        for item in reader:
            rec = accept(item.get(field), "", REFUSAL, "yue-harmful", "refuse", seen, dropped)
            if rec:
                rows.append(rec)
    return rows, dropped


def reservoir_translate(path: Path, k: int, rng: random.Random) -> list[dict]:
    """从旧翻译 jsonl 抽 k 条，改写成带新 system 的 ChatML。"""
    if k <= 0 or not path.exists():
        return []
    chosen: list[dict] = []
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            n += 1
            item = json.loads(line)
            rec = {
                "instruction": norm_text(item.get("instruction")),
                "input": norm_input(item.get("input")),
                "output": norm_text(item.get("output")),
                "source": "translate-sample",
                "task": "translate",
            }
            if not rec["instruction"] or not rec["output"]:
                continue
            rec["id"] = record_id(rec["instruction"], rec["input"], rec["output"])
            rec["text"] = to_chatml(rec["instruction"], rec["input"], rec["output"])
            if len(chosen) < k:
                chosen.append(rec)
            else:
                j = rng.randrange(n)
                if j < k:
                    chosen[j] = rec
    return chosen


def stratified_split(rows: list[dict], seed: int, val_ratio: float, test_ratio: float):
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    train, val, test = [], [], []
    for source, group in by_source.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, int(n * test_ratio)) if n >= 50 else max(1, n // 20 or (1 if n > 2 else 0))
        n_val = max(1, int(n * val_ratio)) if n >= 50 else max(1, n // 20 or (1 if n > 4 else 0))
        if n_test + n_val >= n:
            n_test = min(n_test, 1)
            n_val = min(n_val, 1 if n > 2 else 0)
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])
        print(f"  split {source}: train={n - n_test - n_val} val={n_val} test={n_test}")
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument(
        "--mix-translate",
        type=int,
        default=8000,
        help="从 cleaned_data/tr/yue_train.jsonl 抽 N 条翻译，0 表示不混",
    )
    parser.add_argument(
        "--translate-file",
        type=Path,
        default=ROOT / "cleaned_data" / "tr" / "yue_train.jsonl",
    )
    args = parser.parse_args()

    seen: set[str] = set()
    loaders = [
        ("yue-alpaca", load_alpaca),
        ("cantonese-dialogue", load_dialogue),
        ("allaspect-qa", load_allaspect),
        ("wiki-doyouknow", load_doyouknow),
        ("yue-harmful", load_refuse),
    ]

    all_rows: list[dict] = []
    source_stats = {}
    for name, fn in loaders:
        rows, dropped = fn(seen)
        all_rows.extend(rows)
        source_stats[name] = dict(dropped)
        print(f"{name}: raw={dropped['raw']} kept={dropped['kept']} drop={dropped['raw'] - dropped['kept']}")
        for key, val in sorted(dropped.items()):
            if key not in {"raw", "kept"} and val:
                print(f"    {key}: {val}")

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    print("按来源分层划分…")
    train, val, test = stratified_split(all_rows, args.seed, args.val_ratio, args.test_ratio)

    mix_train = list(train)
    translate_n = 0
    if args.mix_translate:
        sampled = reservoir_translate(args.translate_file, args.mix_translate, rng)
        translate_n = len(sampled)
        mix_train.extend(sampled)
        rng.shuffle(mix_train)
        print(f"混入翻译样本: {translate_n}（来自 {args.translate_file}）")

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "all.jsonl", all_rows)
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "val.jsonl", val)
    write_jsonl(OUT / "test.jsonl", test)
    write_jsonl(OUT / "mix_train.jsonl", mix_train)
    write_jsonl(OUT / "samples.jsonl", test[:20])

    by_source = Counter(r["source"] for r in all_rows)
    by_task = Counter(r["task"] for r in all_rows)
    stats = {
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "schema": ["id", "instruction", "input", "output", "source", "task", "text"],
        "filters": {
            "min_instruction": MIN_INST,
            "max_instruction": MAX_INST,
            "min_output": MIN_OUT,
            "max_output": MAX_OUT,
            "require_cjk": True,
            "dedup": "sha1(instruction+input+output)",
        },
        "sources": source_stats,
        "kept_by_source": dict(by_source),
        "kept_by_task": dict(by_task),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "all": len(all_rows),
            "mix_train": len(mix_train),
            "translate_in_mix": translate_n,
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
