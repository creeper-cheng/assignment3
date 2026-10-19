# -*- coding: utf-8 -*-
r"""把「10 道评测题」和「一页结论」排进一个 PDF。

数据全部来自已经跑完的产物，不重新调 API：
    questions.json      题目与标准答案
    out/results.json    判定、召回覆盖、模型回答
    结论.md             一页结论

渲染走 Playwright 的本机 Edge（和截图同一个浏览器），中文不用额外装字体。

用法：
    python rag\eval\make_pdf.py
    python rag\eval\make_pdf.py --out 第三周作业_题目与结论.pdf
"""
import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = HERE / "out"

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
  font-size: 10.5pt; line-height: 1.65; color: #23213a; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 4px; color: #3b2d7a; letter-spacing: .5px; }
h2 { font-size: 14pt; margin: 22px 0 8px; color: #3b2d7a;
     border-left: 4px solid #6b5bd2; padding-left: 9px; }
h3 { font-size: 11.5pt; margin: 16px 0 5px; color: #2f2b45; }
.sub { color: #6b6885; font-size: 9.5pt; margin-bottom: 14px; }
.meta { font-size: 9pt; color: #55526e; background: #f6f4ff; border: 1px solid #e2ddf7;
        border-radius: 7px; padding: 8px 11px; margin: 10px 0 16px; }
.meta b { color: #3b2d7a; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 9pt; }
th { background: #efeafd; color: #3b2d7a; text-align: left; font-weight: 600; }
th, td { border: 1px solid #ddd8f0; padding: 5px 7px; vertical-align: top; }
tr:nth-child(even) td { background: #fbfaff; }
.q { margin: 13px 0 15px; padding: 10px 12px; border: 1px solid #e5e1f5;
     border-radius: 8px; page-break-inside: avoid; }
.q .head { font-weight: 600; color: #3b2d7a; margin-bottom: 4px; }
.q .ask { font-size: 10.5pt; margin: 3px 0 6px; }
.q .row { font-size: 9pt; color: #4a4763; margin: 2px 0; }
.q .row b { color: #23213a; }
.pass { color: #0a7d43; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
.ref  { color: #8a6520; font-weight: 600; }
.page-break { page-break-before: always; }
blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid #cfc7ef;
             background: #faf9ff; color: #47435f; }
code { background: #f1eefb; padding: 1px 4px; border-radius: 4px; font-size: 9pt; }
ul, ol { margin: 6px 0 10px; padding-left: 22px; }
li { margin: 3px 0; }
.foot { margin-top: 18px; font-size: 8.5pt; color: #8b88a8; text-align: center; }
"""

VERDICT_CLASS = {
    "PASS": "pass", "REFUSE_OK": "ref", "FAIL": "fail",
}
VERDICT_LABEL = {
    "PASS": "✅ 通过", "REFUSE_OK": "✅ 正确拒答", "FAIL": "❌ 未通过",
}
KIND_LABEL = {"single": "单公司", "cross": "跨公司", "negative": "越界"}


def esc(s):
    return html.escape(str(s or ""))


def md_to_html(text):
    from markdown_it import MarkdownIt
    # breaks=True：结论里「语料/索引/生成」是三行并列，默认会被并成一段
    md = MarkdownIt("default", {"html": False, "linkify": False, "breaks": True})
    return md.render(text)


def build_html(questions, results, conclusion_md):
    by_id = {r["id"]: r for r in results}
    rows = []
    stats = {"PASS": 0, "REFUSE_OK": 0, "FAIL": 0}
    agg_stats = {"PASS": 0, "REFUSE_OK": 0, "FAIL": 0}
    for q in questions:
        r = by_id.get(q["id"], {})
        v = r.get("verdict", "—")
        stats[v] = stats.get(v, 0) + 1
        a = r.get("aggregate") or {}
        av = a.get("verdict")
        if av:
            agg_stats[av] = agg_stats.get(av, 0) + 1
        rows.append(
            f"<tr><td>{esc(q['id'])}</td><td>{esc(KIND_LABEL.get(q['kind'], q['kind']))}</td>"
            f"<td>{esc(q['aspect'])}</td>"
            f"<td class='{VERDICT_CLASS.get(v, '')}'>{esc(VERDICT_LABEL.get(v, v))}</td>"
            f"<td>{r.get('n_companies_recalled', '—')}/16</td>"
            f"<td class='{VERDICT_CLASS.get(av, '')}'>{esc(VERDICT_LABEL.get(av, av or '—'))}</td></tr>")

    detail = []
    for q in questions:
        r = by_id.get(q["id"], {})
        v = r.get("verdict", "—")
        gold = (q.get("gold") or {})
        a = r.get("aggregate") or {}
        # 模型回答里带着 markdown 的 **加粗** 和 - 列表符号，PDF 里按纯文本渲染，
        # 不处理就会原样露出 ** 这种符号
        ans = (r.get("answer") or "").strip().replace("**", "")
        ans = ans[:520] + ("…" if len(ans) > 520 else "")
        parts = [
            "<div class='q'>",
            f"<div class='head'>{esc(q['id'])} · {esc(KIND_LABEL.get(q['kind'], q['kind']))}"
            f" · {esc(q['aspect'])}"
            f" &nbsp; <span class='{VERDICT_CLASS.get(v, '')}'>{esc(VERDICT_LABEL.get(v, v))}</span></div>",
            f"<div class='ask'>{esc(q['q'])}</div>",
            f"<div class='row'><b>标准答案</b>：{esc(gold.get('note'))}</div>",
            f"<div class='row'><b>出处</b>：{esc(gold.get('source'))}</div>",
        ]
        if r.get("reason"):
            parts.append(f"<div class='row'><b>判定依据</b>：{esc(r['reason'])}</div>")
        if a:
            parts.append(
                f"<div class='row'><b>逐家扫描</b>：扫 {a.get('n_company')} 家，"
                f"判定相关 {a.get('n_relevant')} 家 → "
                f"<span class='{VERDICT_CLASS.get(a.get('verdict'), '')}'>"
                f"{esc(VERDICT_LABEL.get(a.get('verdict'), a.get('verdict')))}</span></div>")
        if ans:
            parts.append(f"<div class='row'><b>模型回答（节选）</b>：{esc(ans)}</div>")
        parts.append("</div>")
        detail.append("".join(parts))

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>第三周作业 · 题目与结论</title>
<style>{CSS}</style></head><body>
<h1>第三周作业 · 年报问答知识库</h1>
<div class="sub">题目清单 · 逐题判定 · 一页结论</div>
<div class="meta">
<b>语料</b>：合成纤维制造行业 16 家上市公司 × 2025 年年度报告，7,377 块（切块带公司 / 章节 / 页码）<br>
<b>索引</b>：BAAI/bge-small-zh-v1.5（512 维，fp32）+ BM25，RRF 融合，k=20<br>
<b>生成</b>：deepseek-chat，答案带 [n] 引用块<br>
<b>评测时间</b>：{esc((results[0].get('_run_at') if results else '') or '')}
</div>

<h2>10 道题 · 总览</h2>
<table>
<thead><tr><th>题号</th><th>类型</th><th>考点</th><th>top-k</th><th>召回覆盖</th><th>逐家扫描</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<p style="font-size:9.5pt;color:#4a4763">
合计：top-k <b>{stats.get('PASS',0)}</b> 答对 + <b>{stats.get('REFUSE_OK',0)}</b> 正确拒答
= <b>{stats.get('PASS',0)+stats.get('REFUSE_OK',0)}/10</b>；
逐家扫描（map-reduce，仅 4 道跨公司题）<b>{agg_stats.get('PASS',0)+agg_stats.get('REFUSE_OK',0)}/4</b>。
</p>

<h2>逐题明细</h2>
{''.join(detail)}

<div class="page-break"></div>
{md_to_html(conclusion_md)}
<div class="foot">生成于 {esc(__import__('time').strftime('%Y-%m-%d %H:%M'))}</div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "第三周作业_附录_题目与逐题记录.pdf"))
    ap.add_argument("--html", default=None, help="同时保留中间 HTML，便于排查排版")
    args = ap.parse_args()

    cfg = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    data = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    results = data["results"]
    run_at = (data.get("meta") or {}).get("run_at", "")
    for r in results:
        r["_run_at"] = run_at
    conclusion = (HERE / "结论.md").read_text(encoding="utf-8")

    doc = build_html(cfg["questions"], results, conclusion)
    if args.html:
        Path(args.html).write_text(doc, encoding="utf-8")
        print(f"[html] {args.html}")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(**kw)
                break
            except Exception:
                continue
        if browser is None:
            print("没有可用浏览器"); return 1
        page = browser.new_page()
        page.set_content(doc, wait_until="load")
        page.wait_for_timeout(600)
        page.pdf(path=args.out, format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"})
        browser.close()

    p_out = Path(args.out)
    print(f"[pdf] {p_out}  ({p_out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
