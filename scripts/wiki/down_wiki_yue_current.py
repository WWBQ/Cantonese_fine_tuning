#!/usr/bin/env python3
"""
下载粵文維基百科（zh-yue.wikipedia.org）新版正文导出：
mediawiki_content_current / zh_yuewiki

官方说明: https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports
数据目录: https://dumps.wikimedia.org/other/mediawiki_content_current/zh_yuewiki/

与 down_wiki.py（legacy pages-articles.xml.bz2）不同，本脚本下载按月快照、
可能多分片的 XML bzip2，并在 SHA256SUMS 就绪后批量下载与校验。

示例:
    python3 down_wiki_yue_current.py
    python3 down_wiki_yue_current.py --date 2026-05-01
    python3 down_wiki_yue_current.py --save-dir ./data/zh_yuewiki_current --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

WIKI_ID = "zh_yuewiki"
DUMP_BASE = f"https://dumps.wikimedia.org/other/mediawiki_content_current/{WIKI_ID}"
DATE_DIR_RE = re.compile(r'href="(\d{4}-\d{2}-\d{2})/"')

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = _SCRIPT_DIR.parent / "data"


@dataclass
class FileResult:
    name: str
    status: str  # success | skipped | failed
    size: str = ""
    error: str = ""


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def fetch_text(url: str, timeout: int = 60) -> str:
    """小文件用 curl 拉取（目录页、SHA256SUMS）。"""
    result = subprocess.run(
        ["curl", "-sL", "-f", "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        timeout=timeout + 15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"请求失败 (HTTP {result.returncode}): {url}\n{result.stderr.strip()}"
        )
    return result.stdout


def list_available_dates() -> List[str]:
    html = fetch_text(f"{DUMP_BASE}/")
    dates = DATE_DIR_RE.findall(html)
    return sorted(set(dates), reverse=True)


def bzip2_dir_url(date: str) -> str:
    return f"{DUMP_BASE}/{date}/xml/bzip2/"


def sha256sums_url(date: str) -> str:
    return f"{bzip2_dir_url(date)}SHA256SUMS"


def dump_is_ready(date: str) -> bool:
    result = subprocess.run(
        ["curl", "-sL", "-f", "-o", "/dev/null", "-w", "%{http_code}", sha256sums_url(date)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode != 0:
        return False
    code = (result.stdout or "").strip()
    return code == "200"


def resolve_date(explicit: Optional[str]) -> str:
    if explicit:
        if not dump_is_ready(explicit):
            raise SystemExit(
                f"指定日期 {explicit} 的 SHA256SUMS 尚不可用，请稍后重试或换 --date。\n"
                f"检查: {sha256sums_url(explicit)}"
            )
        return explicit

    for date in list_available_dates():
        if dump_is_ready(date):
            print(f"自动选用最新可用快照: {date}")
            return date
    raise SystemExit("未找到任何已完成的 zh_yuewiki content_current 导出（无 SHA256SUMS）。")


def parse_sha256sums(content: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts
        name = name.lstrip("*").strip()
        if name:
            entries.append((digest, name))
    return entries


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_checksums(save_dir: Path, entries: List[Tuple[str, str]]) -> bool:
    ok = True
    for expected, name in entries:
        path = save_dir / name
        if not path.is_file():
            print(f"  ❌ 校验失败，文件不存在: {name}")
            ok = False
            continue
        actual = sha256_file(path)
        if actual != expected:
            print(f"  ❌ 校验失败: {name}")
            print(f"     期望 {expected}")
            print(f"     实际 {actual}")
            ok = False
        else:
            print(f"  ✅ {name}")
    return ok


def download_one(
    url: str,
    output: Path,
    retries: int,
    timeout_sec: int,
) -> FileResult:
    name = output.name
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file() and output.stat().st_size > 0:
        return FileResult(name, "skipped", format_size(output.stat().st_size))

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [
                    "wget",
                    "-c",
                    "-q",
                    "--timeout=120",
                    "--tries=1",
                    "-O",
                    str(output),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if result.returncode == 0 and output.is_file() and output.stat().st_size > 0:
                return FileResult(name, "success", format_size(output.stat().st_size))
        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            return FileResult(
                name,
                "failed",
                error="未找到 wget，请先安装: brew install wget",
            )
        except Exception as e:
            if attempt == retries:
                if output.is_file() and output.stat().st_size == 0:
                    output.unlink(missing_ok=True)
                return FileResult(name, "failed", error=str(e))

    if output.is_file() and output.stat().st_size == 0:
        output.unlink(missing_ok=True)
    return FileResult(name, "failed", error="重试次数已用尽")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载粵文维基 mediawiki_content_current 正文（新版 XML 导出）",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="快照日期 YYYY-MM-DD（默认: 自动选最新且 SHA256SUMS 已就绪的月份）",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default='./data',
        help=f"保存根目录（默认: {DEFAULT_SAVE_DIR}）",
    )
    parser.add_argument("--workers", type=int, default=2, help="并发下载数（默认 2）")
    parser.add_argument("--retries", type=int, default=3, help="单文件重试次数")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="单文件 wget 超时秒数（默认 3600）",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="下载后不做 SHA256 校验",
    )
    args = parser.parse_args()

    date = resolve_date(args.date)
    save_dir = Path(args.save_dir).resolve() / date
    base_url = bzip2_dir_url(date)

    print("=" * 60)
    print("  粵文维基 mediawiki_content_current 下载")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  wiki_id:  {WIKI_ID}  (zh-yue.wikipedia.org)")
    print(f"  快照日期: {date}")
    print(f"  保存目录: {save_dir}")
    print(f"  源 URL:   {base_url}")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    sums_content = fetch_text(sha256sums_url(date))
    (save_dir / "SHA256SUMS").write_text(sums_content, encoding="utf-8")
    entries = parse_sha256sums(sums_content)
    if not entries:
        raise SystemExit("SHA256SUMS 为空或无法解析，请检查导出是否完成。")

    print(f"\n共 {len(entries)} 个分片文件待下载:\n")
    for _, name in entries:
        print(f"  - {name}")

    results: List[FileResult] = []
    tasks = [(f"{base_url}{name}", save_dir / name) for _, name in entries]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one, url, path, args.retries, args.timeout
            ): path.name
            for url, path in tasks
        }
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            icon = {"success": "✅", "skipped": "⏭ ", "failed": "❌"}[res.status]
            extra = f" ({res.size})" if res.size else ""
            err = f" — {res.error}" if res.error else ""
            print(f"[{done}/{total}] {icon} {res.name}{extra}{err}")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print("\n下载失败，请检查网络后重试（wget -c 支持断点续传）。")
        sys.exit(1)

    if not args.skip_verify:
        print("\nSHA256 校验:")
        if not verify_checksums(save_dir, entries):
            print("\n校验未通过，请删除损坏文件后重新运行本脚本。")
            sys.exit(1)

    total_bytes = sum(f.stat().st_size for f in save_dir.glob("*.xml.bz2"))
    print(f"\n完成。XML 分片合计: {format_size(total_bytes)}")
    print(f"目录: {save_dir}")
    print("\n后续可用 bzcat 解压查看，或交给 wiki XML 解析脚本处理。")
    print("示例: bzcat zh_yuewiki-*.xml.bz2 | head")


if __name__ == "__main__":
    main()
