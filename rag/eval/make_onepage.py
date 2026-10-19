# -*- coding: utf-8 -*-
r"""第三周作业 · 独立的一页结论 PDF。

按第三课课件压缩：
  p61–63 行业全景问题五条路线（各自只覆盖几家、混进异业公司、精炼问题不解决覆盖）
  p90–92 五阶段四工具 / 图谱三个局限（覆盖不到·数字丢口径·入口靠匹配）/ 证据义务 Agent 六步
  p163   作业 A 的交付要求：一页结论——哪类题答得好、哪类翻车、为什么

数据直接读 rag/eval/out/results.json，不重跑任何 API。

用法：
    python rag\eval\make_onepage.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = HERE / "out"

CSS = """
@page { size: A4; margin: 11mm 12mm 10mm 12mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;
       font-size: 8.6pt; line-height: 1.5; color: #23213a; margin: 0; }
h1 { font-size: 15.5pt; margin: 0 0 3px; color: #3b2d7a; letter-spacing: .3px; }
.meta { font-size: 7.8pt; color: #5d5a78; border-left: 3px solid #a83a2e;
        padding: 3px 0 3px 8px; margin: 5px 0 9px; line-height: 1.55; }
h2 { font-size: 10pt; margin: 11px 0 4px; color: #3b2d7a;
     border-left: 4px solid #a83a2e; padding-left: 7px; }
table { width: 100%; border-collapse: collapse; font-size: 7.9pt; margin: 3px 0 6px; }
th { background: #efeafd; color: #3b2d7a; text-align: left; font-weight: 600;
     border: 1px solid #d8d2ee; padding: 3px 5px; }
td { border: 1px solid #e2ddf0; padding: 3px 5px; vertical-align: top; }
tr:nth-child(even) td { background: #fbfaff; }
.ok { color: #0a7d43; font-weight: 700; }
.no { color: #b3261e; font-weight: 700; }
.mid { color: #8a6520; font-weight: 700; }
ul { margin: 3px 0 6px; padding-left: 16px; }
li { margin: 2.5px 0; }
.k { font-weight: 700; color: #23213a; }
.note { background: #f7f4ff; border: 1px solid #e2ddf0; border-radius: 5px;
        padding: 6px 9px; font-size: 8.2pt; margin: 6px 0; line-height: 1.55; }
.foot { margin-top: 8px; font-size: 7.4pt; color: #8b88a8; text-align: center; }
"""


def main():
    data = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    rs = {r["id"]: r for r in data["results"]}
    meta = data.get("meta") or {}

    n_ok = sum(1 for r in data["results"] if r["verdict"] in ("PASS", "REFUSE_OK"))
    aggs = [r["aggregate"] for r in data["results"] if r.get("aggregate")]
    n_agg = sum(1 for a in aggs if a["verdict"] in ("PASS", "REFUSE_OK"))

    q08 = rs.get("q08", {})
    agg08 = q08.get("aggregate") or {}
    rel08 = agg08.get("n_relevant", "—")
    n08 = agg08.get("n_company", "—")

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>一页结论</title><style>{CSS}</style></head><body>

<h1>一页结论：哪类题答得好、哪类翻车、为什么</h1>
<div class="meta">
<b>语料</b> 合成纤维制造 16 家 × 2025 年报，7,377 块（带公司/章节/页码）
<b>索引</b> 向量（bge-small-zh，512 维）+ BM25，RRF 融合，k=20
<b>生成</b> deepseek-chat，答案带 [n] 引用<br>
<b>结果</b> top-k <b>{n_ok}/10</b>（含 1 次正确拒答）；跨公司题逐家扫描 <b>{n_agg}/{len(aggs)}</b>
<b>记录</b> 错误与负面结果全部保留，逐题证据见附录
</div>

<h2>一、逐题结果与归因</h2>
<table>
<tr><th style="width:19%">题型</th><th style="width:11%">结果</th><th style="width:40%">为什么</th><th style="width:30%">下一步</th></tr>
<tr>
  <td>单公司 · 表内数字<br>q01 / q03 / q04</td>
  <td class="ok">3 / 3 通过</td>
  <td>公司名与指标名同时命中 BM25 与向量，第 2 节「主要会计数据」稳定进 top-5，数字与原文逐位一致。</td>
  <td><b>search + read 已够</b>，不必加复杂度。</td>
</tr>
<tr>
  <td>单公司 · 跨表勾稽<br>q02</td>
  <td class="no">未通过</td>
  <td>经营现金流 252,382,527.02 抓对；<b>净利润抓成 105,569,879.54</b>（同页相邻的「综合收益总额」），正确值 197,496,121.75。这是<b>口径错误</b>，不是召回错误——检索到了对的表，取错了行。</td>
  <td>加 <b>find_table</b>：定位到具体表、按行名取值，取数前先锁定指标口径。</td>
</tr>
<tr>
  <td>单公司 · 定性<br>q05</td>
  <td class="no">未通过</td>
  <td>top-20 里只有<b>新凤鸣</b>的「反内卷」段落，桐昆自己的第 3 节 P14 没被召回；模型如实指出「片段出自新凤鸣，并非桐昆」后拒答——<b>召回不全，不是幻觉</b>。</td>
  <td>按实体定向检索，或提高 k；跨主题时走图谱边。</td>
</tr>
<tr>
  <td>跨公司 · 全景<br>q06 / q07 / q08 / q09</td>
  <td class="mid">top-k 0/4<br>逐家扫描 2/4</td>
  <td>top-20 平摊到 16 家，每家仅 1~2 块，q06 只覆盖 6/16、q08 7/16，模型只能拒答。<br>
      <b>q08 逐家扫描同样失败</b>：map 阶段拿<b>同一个全局问题</b>去问每一家，单家语境召回不到第 2 节那张表，16 家里只有 {rel08}/{n08} 家被判相关。<br>
      q09 逐家扫描三家全中——精确关键词题上 BM25 明显强于向量。</td>
  <td>先把全局题<b>拆成按实体的证据义务</b>（每家一个子问题），逐家取证并画覆盖地图；而不是把全局问句丢给每家的召回。</td>
</tr>
<tr>
  <td>越界<br>q10</td>
  <td class="ok">正确拒答</td>
  <td>明确说明语料只有 2025 年，<b>没有</b>拿同表的 2023 年列（61,468,601,763.01）冒充 2022 年数据。</td>
  <td>保留：缺据就说缺据，比编一个像样的数好。</td>
</tr>
</table>

<h2>二、与课件的对应</h2>
<div class="note">
<b>跨公司全景题（课件 p61–63）</b>：五条路线召回的始终只是几家公司的几段话，且各不相同；BM25 会被「行业」「问题」「措施」这类通用词带偏，把做包装、做酒店的公司也召回；把问题精炼一遍也不解决覆盖面。
本项目复现了同一条边界——<b>碎块检索的天花板不在排序，在覆盖</b>。<br>
<b>数字丢口径（课件 p91）</b>：数值离开原表就只剩下一个数，是谁的、哪个口径、哪一期必须回到那张表。q02 正是这个坑。
</div>

<h2>三、下一步的方法选择</h2>
<ul>
<li><span class="k">先判问题形状再检索</span>：局部（单公司单指标）／数字（要回表）／全局（跨 N 家）／多跳，四类走不同路径，不写死在同一套流程里。</li>
<li><span class="k">数字题加表格定位</span>：find_table 命中「第 2 节 主要会计数据」，按行名取值并保留口径标签（归母净利润 / 净利润 / 综合收益总额），避免 q02 式误取。</li>
<li><span class="k">全局题按实体拆义务</span>：把「哪几家亏损」拆成 16 条子义务，逐家提问、逐家取证，用覆盖地图标「已覆盖／部分／缺据」，缺哪家补哪家——这直接修掉 q08。</li>
<li><span class="k">证据白名单 + 缺据明说</span>：最终上下文只放真正读过的原文块；引用不存在的 ID 直接拒绝；不设固定轮数，工具连续不再带回新证据就停。</li>
</ul>

<div class="foot">
附录：6 页《题目与逐题记录》（含全部召回块、引用与判定依据，错误与负面结果保留）　·
索引 {meta.get('n','—')} 块 / {meta.get('n_pdf','—')} 份 / dim {meta.get('dim','—')}
</div>
</body></html>"""

    from playwright.sync_api import sync_playwright
    out = REPO / "第三周作业_一页结论.pdf"
    with sync_playwright() as p:
        b = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                b = p.chromium.launch(**kw)
                break
            except Exception:
                continue
        if b is None:
            print("没有可用浏览器"); return 1
        page = b.new_page()
        page.set_content(html, wait_until="load")
        page.wait_for_timeout(400)
        page.pdf(path=str(out), format="A4", print_background=True,
                 margin={"top": "11mm", "bottom": "10mm", "left": "12mm", "right": "12mm"})
        b.close()

    import pypdfium2 as pdfium
    n = len(pdfium.PdfDocument(str(out)))
    print(f"[pdf] {out}  {out.stat().st_size/1024:.0f} KB  页数={n}")
    if n != 1:
        print("!! 超过一页，需要再压缩")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
