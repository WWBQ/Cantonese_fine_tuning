#!/usr/bin/env python3
"""同一套 system + 同一批题，对比基座 / 旧翻译 LoRA / 新问答 LoRA。

本机 Ollama：
  python3 scripts/compare_models.py --backend ollama \\
    --base qwen2.5:7b --old yue-translator --ft yue_qwen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SET_PATH = ROOT / "eval" / "compare_set.jsonl"

SYSTEM = (
    "你是一个粤语助手。用粤语回答问题、完成写作或对话。"
    "只有用户明确要求翻译时才翻译。不要把普通提问当成翻译任务。"
)

TRANSLATE_RE = re.compile(r"(请将|請將|翻译成|翻譯成|译成|譯成)")


def load_set(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def user_text(row: dict) -> str:
    inst = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return f"{inst}\n\n{inp}" if inp else inst


def looks_unsolicited_translate(prompt: str, output: str) -> bool:
    if TRANSLATE_RE.search(prompt):
        return False
    out = (output or "").strip()
    if len(out) < 8 or len(out) >= 120:
        return False
    return len(out) <= len(prompt) + 40


def asked_translate(prompt: str) -> bool:
    return bool(TRANSLATE_RE.search(prompt))


class OllamaRunner:
    def __init__(self, model: str, host: str):
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"

    def generate(self, user: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model,
            "system": SYSTEM,
            "prompt": user,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_new_tokens,
            },
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()


class UnslothRunner:
    def __init__(self, model_name: str):
        os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from contextlib import contextmanager

        import torch
        import torch._higher_order_ops.utils as _hou
        from unsloth import FastLanguageModel

        if not hasattr(_hou, "setup_compilation_env"):
            @contextmanager
            def _setup_compilation_env(*_a, **_k):
                yield

            _hou.setup_compilation_env = _setup_compilation_env

        self.torch = torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, user: str, max_new_tokens: int) -> str:
        prompt = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text.strip()


def safe_gen(runner, user: str, max_new_tokens: int) -> str:
    try:
        return runner.generate(user, max_new_tokens)
    except Exception as exc:
        return f"[ERROR] {exc}"


def write_report(rows: list[dict], path: Path, has_old: bool) -> None:
    n = len(rows)
    keys = ["base"] + (["old"] if has_old else []) + ["ft"]
    labels = {"base": "基座", "old": "旧翻译 LoRA", "ft": "新问答 LoRA"}
    header = "| |" + "".join(f" {labels[k]} |" for k in keys)
    sep = "|---|" + "---:|" * len(keys)
    uns_row = "| 未要求翻译、回复却像短译 |" + "".join(
        f" {sum(1 for r in rows if r.get(k + '_unsolicited_translate'))} |" for k in keys
    )
    trans_n = sum(1 for r in rows if r.get("asked_translate"))
    trans_row = "| 明确要求翻译的题 |" + "".join(f" {trans_n} |" for _ in keys)

    lines = [
        "# 基座 vs 旧翻译 LoRA vs 新问答 LoRA",
        "",
        f"共 {n} 条。三套模型同一 system：「粤语助手，未要求则不翻译」。",
        "",
        header,
        sep,
        uns_row,
        trans_row,
        "",
        "「像短译」只是粗规则。旧翻译 LoRA 若在同一 system 下仍把提问拿去翻译，说明数据把行为训死了。",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['id']} · {r['task']} / {r.get('bucket')}")
        lines.append("")
        lines.append(f"**用户:** {r['prompt']}")
        if r.get("expect"):
            lines.append("")
            lines.append(f"**期望:** {r['expect']}")
        for k in keys:
            lines.append("")
            lines.append(f"**{labels[k]}:** {r.get(k, '')}")
        if r.get("gold"):
            lines.append("")
            lines.append(f"**参考答案:** {r['gold'][:400]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, default=SET_PATH)
    parser.add_argument("--backend", choices=["ollama", "unsloth"], default="ollama")
    parser.add_argument("--base", default="qwen2.5:7b")
    parser.add_argument("--old", default="", help="旧翻译模型，Ollama 名或路径；空则不跑")
    parser.add_argument("--ft", default="yue_qwen")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_set(args.set)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit(f"空对比集: {args.set}")

    make = (
        (lambda name: OllamaRunner(name, args.host))
        if args.backend == "ollama"
        else UnslothRunner
    )
    runners = {"base": make(args.base), "ft": make(args.ft)}
    if args.old:
        runners["old"] = make(args.old)

    out_jsonl = ROOT / "eval" / "compare_results.jsonl"
    out_jsonl.write_text("", encoding="utf-8")
    results = []
    order = ["base"] + (["old"] if "old" in runners else []) + ["ft"]
    for i, row in enumerate(rows, 1):
        prompt = user_text(row)
        print(f"[{i}/{len(rows)}] {row['id']}", flush=True)
        rec = {
            **{k: row.get(k) for k in ("id", "bucket", "task", "source", "expect", "gold")},
            "prompt": prompt,
            "asked_translate": asked_translate(prompt),
        }
        for key in order:
            text = safe_gen(runners[key], prompt, args.max_new_tokens)
            rec[key] = text
            rec[key + "_unsolicited_translate"] = looks_unsolicited_translate(prompt, text)
        results.append(rec)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = ROOT / "eval" / "compare_report.md"
    write_report(results, report, has_old="old" in runners)
    print(f"jsonl: {out_jsonl}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
