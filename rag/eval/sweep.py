# -*- coding: utf-8 -*-
r"""逐家扫描：给跨公司题建「可核对的标准答案」。

跨公司题的 top-k 召回天然盖不满 16 家，所以标准答案不能从一次检索里取，
得每家单独问一遍、把命中的原文块dump出来，人工核对后再回填 questions.json。

用法：
    python rag\eval\sweep.py --field "2025年归属于上市公司股东的净利润" \
        --out rag\eval\out\sweep_净利润.md
    python rag\eval\sweep.py --field "2025年存货跌价准备" --k 4 --truncate 900
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "rag"))
from ragcli import Client  # noqa: E402


def setup_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8799")
    ap.add_argument("--field", required=True, help="要扫的字段/问题，会拼成「公司名 + field」")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--truncate", type=int, default=1100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(REPO / "rag" / "corpus_2025.json"))
    args = ap.parse_args()

    cfg = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    year = cfg.get("report_year", 2025)
    c = Client(args.base)
    h = c.health()
    if not h.get("index_ready"):
        print("索引未就绪"); return 1
    print(f"[索引] n={h['n']} 份={h.get('n_pdf')} 范围={h.get('scope')}")

    lines = [f"# 逐家扫描：{args.field}", "",
             f"索引 {h['n']} 块 / {h.get('n_pdf')} 份 · k={args.k}", ""]
    for i, co in enumerate(cfg["companies"], 1):
        q = f"{co['name']}{year}年{args.field}"
        r = c.retrieve(q, k=args.k)
        nbs = r["neighbors"]
        own = [n for n in nbs if n["company"] == co["name"]]
        lines += [f"## {i:02d}. {co['name']}（{co['code']}）· {co['segment']}", "",
                  f"查询：`{q}`　返回 {len(nbs)} 块，其中本公司 {len(own)} 块", ""]
        print(f"[{i:02d}/16] {co['name']:<8} 返回 {len(nbs)} 块 / 本公司 {len(own)} 块")
        for n in nbs:
            d = c.get(f"/api/chunk?row={n['row']}")
            txt = (d.get("text") or "")[:args.truncate]
            flag = "" if n["company"] == co["name"] else "  ⚠ 别家公司"
            lines += [f"**[{n['rank']}] {n['company']} · {n['chapter']} · P{n['page']}**{flag}",
                      f"RRF={n['rrf']:.4f} 余弦={n['sim']:.4f}(#{n.get('vrank')}) "
                      f"BM25={n.get('bm25', 0):.1f}(#{n.get('brank')})", "",
                      "```", txt.replace("```", "'''"), "```", ""]
        lines.append("---\n")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[完成] → {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
