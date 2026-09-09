#!/usr/bin/env python3
"""
Wikimedia 全项目全语言批量下载工具
下载 7 个 Wikimedia 项目（Wikisource + Wiktionary + Wikibooks + Wikiquote + Wikinews + Wikivoyage + Wikiversity）
的所有语言版本 pages-articles dump 文件。

功能特性：
- 断点续传（基于文件是否已存在且非空）
- 多线程并发下载（默认 4 线程）
- 自动重试（默认 3 次）
- 下载进度显示
- 失败汇总与重试

使用方式：
    python3 download_all_wikimedia.py [--save-dir /path/to/save] [--workers 4] [--retries 3]
    python3 download_all_wikimedia.py --projects wikisource wiktionary  # 只下载指定项目
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================
# 各项目的语言列表
# ============================================================

PROJECTS: Dict[str, List[str]] = {
    "wikisource": [
        "ang", "ar", "as", "az", "ban", "bcl", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy",
        "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu",
        "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "kn", "ko",
        "la", "lij", "li", "lt", "mad", "min", "mk", "ml", "mr", "ms", "my", "nan", "nap",
        "nl", "no", "or", "pa", "pl", "pms", "pt", "ro", "ru", "sah", "sa", "sk", "sl", "sr",
        "su", "sv", "ta", "tcy", "te", "th", "tl", "tr", "uk", "ur", "vec", "vi", "wa", "yi", "zh",
    ],
    "wiktionary": [
        "aa", "ab", "af", "ak", "am", "ang", "an", "ar", "ast", "as", "av", "ay", "az",
        "bcl", "be", "bew", "bg", "bh", "bi", "bjn", "blk", "bm", "bn", "bo", "br", "bs", "btm",
        "ca", "chr", "ch", "ckb", "co", "cr", "csb", "cs", "cy",
        "da", "de", "diq", "dv", "dz",
        "el", "en", "eo", "es", "et", "eu",
        "fa", "fi", "fj", "fo", "fr", "fy",
        "ga", "gd", "gl", "gn", "gom", "gor", "gu", "guw", "gv",
        "ha", "he", "hif", "hi", "hr", "hsb", "hu", "hy",
        "ia", "id", "ie", "ig", "ik", "io", "is", "it", "iu",
        "ja", "jbo", "jv",
        "kaa", "ka", "kbd", "kcg", "kk", "kl", "km", "kn", "ko", "ks", "ku", "kw", "ky",
        "la", "lb", "li", "lmo", "ln", "lo", "lt", "lv",
        "mad", "mg", "mh", "min", "mi", "mk", "ml", "mni", "mn", "mnw", "mr", "ms", "mt", "my",
        "nah", "nan", "na", "nds", "ne", "nia", "nl", "nn", "no",
        "oc", "om", "or",
        "pa", "pi", "pl", "pnb", "ps", "pt",
        "qu",
        "rm", "rn", "ro", "rup", "ru", "rw",
        "sat", "sa", "scn", "sc", "sd", "sg", "shn", "sh", "shy", "simple", "si", "skr",
        "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw",
        "ta", "tcy", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tpi", "tr", "ts", "tt", "tw",
        "ug", "uk", "ur", "uz",
        "vec", "vi", "vo",
        "wa", "wo",
        "xh",
        "yi", "yo", "yue",
        "za", "zgh", "zh", "zu",
    ],
    "wikibooks": [
        "aa", "af", "ak", "ang", "ar", "ast", "as", "ay", "az",
        "ba", "be", "bg", "bi", "bm", "bn", "bo", "bs",
        "ca", "ch", "co", "cs", "cv", "cy",
        "da", "de",
        "el", "en", "eo", "es", "et", "eu",
        "fa", "fi", "fr", "fy",
        "ga", "gl", "gn", "got", "gu",
        "he", "hi", "hr", "hu", "hy",
        "ia", "id", "ie", "is", "it",
        "ja",
        "ka", "kk", "km", "kn", "ko", "ks", "ku", "ky",
        "la", "lb", "li", "ln", "lt", "lv",
        "mg", "min", "mi", "mk", "ml", "mn", "mr", "ms", "my",
        "nah", "nan", "na", "nds", "ne", "nl", "no",
        "oc",
        "pa", "pl", "ps", "pt",
        "qu",
        "rm", "ro", "ru",
        "sa", "se", "shn", "simple", "si", "sk", "sl", "sq", "sr", "su", "sv", "sw",
        "ta", "te", "tg", "th", "tk", "tl", "tr", "tt",
        "ug", "uk", "ur", "uz",
        "vi", "vo",
        "wa",
        "xh",
        "yo",
        "za", "zh", "zu",
    ],
    "wikiquote": [
        "af", "am", "ang", "ar", "ast", "as", "az",
        "bcl", "be", "bg", "bjn", "bm", "bn", "br", "bs",
        "ca", "co", "cr", "cs", "cy",
        "da", "de",
        "el", "en", "eo", "es", "et", "eu",
        "fa", "fi", "fr",
        "ga", "gl", "gor", "gu", "guw",
        "he", "hi", "hr", "hu", "hy",
        "id", "ig", "is", "it",
        "ja",
        "ka", "kk", "kn", "ko", "kr", "ks", "ku", "kw", "ky",
        "la", "lb", "li", "lt",
        "ml", "mr", "ms",
        "nan", "na", "nds", "nl", "nn", "no",
        "pcm", "pl", "pt",
        "qu",
        "ro", "ru",
        "sah", "sa", "simple", "sk", "sl", "sq", "sr", "su", "sv",
        "ta", "te", "th", "tk", "tl", "tr", "tt",
        "ug", "uk", "ur", "uz",
        "vi", "vo",
        "wo",
        "za", "zh",
    ],
    "wikinews": [
        "ar", "bg", "bs", "ca", "cs", "de", "el", "en", "eo", "es", "fa", "fi", "fr",
        "guw", "he", "hu", "it", "ja", "ko", "li", "nl", "no", "pl", "pt", "ro", "ru",
        "sd", "shn", "sq", "sr", "sv", "ta", "th", "tr", "uk", "zh",
    ],
    "wikivoyage": [
        "bn", "cs", "de", "el", "en", "eo", "es", "fa", "fi", "fr", "he", "hi", "id",
        "it", "ja", "nl", "pl", "ps", "pt", "ro", "ru", "shn", "sv", "tr", "uk", "vi", "zh",
    ],
    "wikiversity": [
        "ar", "beta", "cs", "de", "el", "en", "es", "fi", "fr", "hi", "it", "ja", "ko",
        "pt", "ru", "sl", "sv", "zh",
    ],
}


@dataclass
class DownloadResult:
    """单个下载任务的结果"""
    project: str
    lang: str
    url: str
    output: str
    status: str  # "success" | "skipped" | "failed"
    size: str = ""
    error: str = ""


def get_url(project: str, lang: str) -> str:
    """生成下载 URL"""
    prefix = f"{lang}{project}"
    return f"https://dumps.wikimedia.org/{prefix}/latest/{prefix}-latest-pages-articles.xml.bz2"


def get_output_path(save_base: str, project: str, lang: str) -> str:
    """生成保存路径"""
    prefix = f"{lang}{project}"
    return os.path.join(save_base, project, f"{prefix}-latest-pages-articles.xml.bz2")


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def download_file(project: str, lang: str, save_base: str, retries: int = 3) -> DownloadResult:
    """下载单个文件（使用 wget，支持断点续传和重试）"""
    url = get_url(project, lang)
    output = get_output_path(save_base, project, lang)

    # 确保目录存在
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # 已存在且非空，跳过
    if os.path.isfile(output) and os.path.getsize(output) > 0:
        size = format_size(os.path.getsize(output))
        return DownloadResult(project, lang, url, output, "skipped", size)

    # 使用 wget 下载
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["wget", "-c", "-q", "--timeout=120", f"--tries=1", "-O", output, url],
                capture_output=True,
                text=True,
                timeout=600,  # 单文件最长 10 分钟
            )
            if result.returncode == 0 and os.path.isfile(output) and os.path.getsize(output) > 0:
                size = format_size(os.path.getsize(output))
                return DownloadResult(project, lang, url, output, "success", size)
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            if attempt == retries:
                # 清理空文件
                if os.path.isfile(output) and os.path.getsize(output) == 0:
                    os.remove(output)
                return DownloadResult(project, lang, url, output, "failed", error=str(e))

    # 所有重试都失败
    if os.path.isfile(output) and os.path.getsize(output) == 0:
        os.remove(output)
    return DownloadResult(project, lang, url, output, "failed", error="All retries exhausted")


def print_summary(results: List[DownloadResult], project: Optional[str] = None):
    """打印下载摘要"""
    if project:
        proj_results = [r for r in results if r.project == project]
    else:
        proj_results = results

    success = sum(1 for r in proj_results if r.status == "success")
    skipped = sum(1 for r in proj_results if r.status == "skipped")
    failed = sum(1 for r in proj_results if r.status == "failed")

    title = project if project else "全部项目"
    print(f"\n{'=' * 50}")
    print(f"  {title} 下载统计")
    print(f"  成功: {success}  跳过(已存在): {skipped}  失败: {failed}")
    print(f"{'=' * 50}")

    if failed > 0:
        print(f"\n  失败列表:")
        for r in proj_results:
            if r.status == "failed":
                print(f"    ❌ {r.lang}{r.project}: {r.error}")


def main():
    parser = argparse.ArgumentParser(
        description="Wikimedia 全项目全语言批量下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 下载全部 7 个项目的所有语言
    python3 download_all_wikimedia.py

    # 指定保存目录
    python3 download_all_wikimedia.py --save-dir /data/wikimedia

    # 只下载 wikisource 和 wiktionary
    python3 download_all_wikimedia.py --projects wikisource wiktionary

    # 使用 8 个线程并发下载
    python3 download_all_wikimedia.py --workers 8

    # 只下载指定语言
    python3 download_all_wikimedia.py --projects wikisource --langs zh en ja de fr
        """,
    )
    parser.add_argument(
        "--save-dir", default="/data/workspace/wikimedia_all",
        help="保存根目录（默认: /data/workspace/wikimedia_all）",
    )
    parser.add_argument(
        "--projects", nargs="+", default=None,
        choices=list(PROJECTS.keys()),
        help="要下载的项目列表（默认: 全部 7 个）",
    )
    parser.add_argument(
        "--langs", nargs="+", default=None,
        help="只下载指定语言（默认: 各项目全部语言）",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="并发下载线程数（默认: 4）",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="每个文件的最大重试次数（默认: 3）",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="只重试之前失败的文件（跳过已存在的）",
    )

    args = parser.parse_args()

    # 确定要下载的项目
    projects_to_download = args.projects if args.projects else list(PROJECTS.keys())

    # 构建任务列表
    tasks = []
    for project in projects_to_download:
        langs = args.langs if args.langs else PROJECTS[project]
        for lang in langs:
            tasks.append((project, lang))

    # 打印计划
    print("=" * 60)
    print(f"  Wikimedia 全项目全语言批量下载")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  下载项目: {', '.join(projects_to_download)}")
    print(f"  总任务数: {len(tasks)}")
    print(f"  并发线程: {args.workers}")
    print(f"  重试次数: {args.retries}")
    print("=" * 60)
    print()

    # 按项目分组统计
    for project in projects_to_download:
        count = sum(1 for p, _ in tasks if p == project)
        print(f"  📦 {project}: {count} 个语言")
    print()

    # 开始下载
    all_results: List[DownloadResult] = []
    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(download_file, project, lang, args.save_dir, args.retries): (project, lang)
            for project, lang in tasks
        }

        for future in as_completed(future_to_task):
            result = future.result()
            all_results.append(result)
            completed += 1

            # 进度输出
            icon = {"success": "✅", "skipped": "⏭ ", "failed": "❌"}[result.status]
            size_info = f" ({result.size})" if result.size else ""
            print(
                f"[{completed}/{total}] {icon} {result.lang}{result.project}{size_info}"
            )

    # 按项目打印摘要
    print("\n\n" + "=" * 60)
    print("  📊 下载总结")
    print("=" * 60)

    for project in projects_to_download:
        print_summary(all_results, project)

    # 总体统计
    total_success = sum(1 for r in all_results if r.status == "success")
    total_skipped = sum(1 for r in all_results if r.status == "skipped")
    total_failed = sum(1 for r in all_results if r.status == "failed")

    print(f"\n{'=' * 60}")
    print(f"  🎯 总计: 成功={total_success}  跳过={total_skipped}  失败={total_failed}")
    print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    # 显示各目录大小
    print("\n各项目占用空间:")
    for project in projects_to_download:
        proj_dir = os.path.join(args.save_dir, project)
        if os.path.isdir(proj_dir):
            total_bytes = sum(
                f.stat().st_size for f in Path(proj_dir).glob("*.bz2") if f.is_file()
            )
            print(f"  {project}: {format_size(total_bytes)}")

    total_bytes = sum(
        f.stat().st_size
        for f in Path(args.save_dir).rglob("*.bz2")
        if f.is_file()
    )
    print(f"\n  总计: {format_size(total_bytes)}")

    # 返回退出码
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()