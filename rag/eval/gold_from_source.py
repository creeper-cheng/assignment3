# -*- coding: utf-8 -*-
r"""直接从切块原文里取标准答案 —— 不经检索、不经大模型。

为什么要绕开 RAG 建标准答案：拿检索系统的输出当自己的答案，再去评检索系统，
是循环论证。所以标准答案只认 PDF 抽出来的原文，脚本做的只是「把含关键字段的
段落捞出来给人看」，不做任何推断。

用法：
    python rag\eval\gold_from_source.py --probe "归属于上市公司股东的净利润"
    python rag\eval\gold_from_source.py --probe "存货跌价准备" --companies 000703,601233
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "rag"))
CHUNK_DIR = REPO / "年报_合成纤维制造" / "_derived" / "chunks"


def setup_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="要找的字段名，如「归属于上市公司股东的净利润」")
    ap.add_argument("--companies", default=None, help="逗号分隔的股票代码，默认 corpus_2025.json 全部")
    ap.add_argument("--corpus", default=str(REPO / "rag" / "corpus_2025.json"))
    ap.add_argument("--window", type=int, default=700, help="命中处前后各取多少字")
    ap.add_argument("--per-company", type=int, default=2, help="每家公司最多看几个块")
    ap.add_argument("--require-chapter", default=None, help="限定章节子串，如「第2节」")
    args = ap.parse_args()

    cfg = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    year = cfg.get("report_year", 2025)
    if args.companies:
        want = {s.strip() for s in args.companies.split(",") if s.strip()}
        cos = [c for c in cfg["companies"] if c["code"] in want]
    else:
        cos = cfg["companies"]

    probe = args.probe
    print(f"# 标准答案抽取：{probe}\n")
    print(f"来源：切块原文（年报 PDF 抽取结果），年报期 {year}，"
          f"{len(cos)} 家。**以下为原文片段，未作任何推断。**\n")

    for co in cos:
        pid = f"{co['code']}_{co['name']}_{year}"
        f = CHUNK_DIR / f"{pid}.jsonl"
        if not f.exists():
            print(f"## {co['name']}（{co['code']}）\n\n> 切块文件不存在：{f.name}\n")
            continue
        hits = []
        for ln in open(f, encoding="utf-8"):
            if not ln.strip():
                continue
            c = json.loads(ln)
            if args.require_chapter and args.require_chapter not in (c.get("chapter") or ""):
                continue
            t = c.get("text") or ""
            i = t.find(probe)
            if i < 0:
                continue
            a = max(0, i - args.window // 2)
            b = min(len(t), i + args.window // 2)
            hits.append((c, a, b, t[a:b]))
            if len(hits) >= args.per_company:
                break
        print(f"## {co['name']}（{co['code']}）· {co['segment']}")
        print()
        if not hits:
            print(f"> 原文中**未出现**「{probe}」。\n")
            continue
        for c, a, b, seg in hits:
            print(f"`{c['chapter']}` P{c['page']} · chunk `{c['chunk_id']}`"
                  f"{' · 含表格' if c.get('has_table') else ''}")
            print()
            print("```")
            print(seg)
            print("```")
            print()
        print("---\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
