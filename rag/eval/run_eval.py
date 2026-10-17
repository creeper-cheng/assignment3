# -*- coding: utf-8 -*-
r"""第三周作业 · 逐题评测。

对 questions.json 里的每道题跑完整链路（混合召回 → DeepSeek 生成），
把「召回了哪些块」「引用了哪些块」「答对没有」「错在哪」逐题落盘。

产出（默认 rag/eval/out/）：
    results.json     机器可读的完整记录（每题：邻居全字段、答案、引用、判定）
    evidence/qNN.md  每题 top-k 块的原文，用来人工核对标准答案
    评测记录.md      人读版：逐题表格 + 答案 + 判定

用法：
    python rag\eval\run_eval.py                      # 全部
    python rag\eval\run_eval.py --only q01,q10       # 只跑指定题
    python rag\eval\run_eval.py --no-llm             # 只检索不生成（省时间）
"""
import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "rag"))
from ragcli import Client  # noqa: E402

OUT = HERE / "out"

# 拒绝/越界的正面表述——负向题命中这些才算「守住了」
REFUSE_MARKERS = [
    "未披露", "没有披露", "不在", "未包含", "不包含", "没有收录", "未收录",
    "无法确定", "无法回答", "没有找到", "未找到", "找不到", "无法从", "语料",
    "所给材料", "提供的片段", "无法得知", "不可得", "缺失", "未提供", "没有相关",
]
# 编造的大额数字：负向题里出现 6 位以上连续数字（去掉千分位后）就可疑
MONEY_RE = re.compile(r"\d{6,}")


def setup_console():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def norm(s):
    """归一化：去掉空白、半/全角逗号、千分位，便于数字比对。"""
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s,，]+", "", s)


NUM_ONLY = re.compile(r"^-?[\d,]+(\.\d+)?$")
CLAUSE_SPLIT = re.compile(r"[。；;！!？?\n]")

# 「回答自己说没答出来」的表述。正向题命中这些就等于没回答——
# 关键词打分不看语境会把这种拒答判成通过（q05「反内卷」、q09「神马」都这么骗过第一版判定）。
INSUFFICIENT = ("无法判断", "无法确定", "不能判断", "不足以回答", "无法回答",
                "无法得出", "无法比较", "不足以完整回答", "不能确定")
# 出现在关键词附近就算「否定语境」——不能把「X 未提及 Y」算成「X 说了 Y」
NEG_MARKERS = ("未出现", "没有出现", "未提及", "没有提及", "未披露", "未包含",
               "不包含", "并非", "不是", "未能", "无相关", "未找到", "没有找到",
               "均未", "未提供", "无法", "不足")


def amount_variants(s):
    """把金额展开成回答里可能出现的写法。

    「2032802043.01」在回答里常被写成「20.33亿」或「203,280.20万」，
    只比原串会把对的答案判成错的。所以额外接受按亿/万换算后的两位小数。
    """
    out = []
    if not NUM_ONLY.match(s or ""):
        return out
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return out
    if abs(v) < 1e4:
        return out
    out += [f"{v / 1e8:.2f}", f"{v / 1e4:.2f}"]
    return out


def judge(case, answer):
    """返回 (verdict, reason)。verdict ∈ PASS / FAIL / PENDING / REFUSE_OK / REFUSE_FAIL"""
    gold = case.get("gold") or {}
    if case.get("expect_refusal"):
        hit = [m for m in REFUSE_MARKERS if m in answer]
        nums = MONEY_RE.findall(norm(answer))
        if nums:
            return "REFUSE_FAIL", f"应说明语料不含该年份，却给出了大额数字 {nums[:4]}"
        if hit:
            return "REFUSE_OK", f"正确说明数据缺失（命中「{hit[0]}」）"
        return "REFUSE_FAIL", "既没说数据不在语料里，也没给出数字——表述含糊"
    must = gold.get("must") or []
    if not must:
        return "PENDING", "标准答案未填，需人工核对原文后回填 gold.must"
    na = norm(answer)

    def affirmative(needle_n):
        """needle 是否出现在**非否定**语境里。

        只看命中所处的「小句」，不看向左右固定宽度——map-reduce 的模板会给
        每家都缀一句「产能计划：未披露/片段不足」，固定窗口会把肯定句也误判成否定。
        """
        for m in re.finditer(re.escape(needle_n), na):
            start = 0
            for sep in CLAUSE_SPLIT.finditer(na, 0, m.start()):
                start = sep.end()
            end_m = CLAUSE_SPLIT.search(na, m.end())
            clause = na[start:end_m.start() if end_m else len(na)]
            if not any(x in clause for x in NEG_MARKERS):
                return True
        return False

    miss = []
    for m in must:
        if affirmative(norm(m)):
            continue
        if any(affirmative(v) for v in amount_variants(m)):
            continue
        miss.append(m)
    if miss:
        # 关键信息不全时再看它是不是干脆承认答不了：
        # map-reduce 的模板会给每家缀一句「未披露/片段不足」，所以这条
        # 只在**确实缺料**时才用，不能拿来否决已经答全的回答。
        hit = [p for p in INSUFFICIENT if p in na]
        if hit:
            return "FAIL", f"缺关键信息 {miss}，且自称信息不足（「{hit[0]}」）"
        return "FAIL", f"缺关键信息（未在肯定语境中出现）：{miss}"
    return "PASS", f"关键信息全部命中：{must}"


def dump_evidence(c, qid, q, neighbors, evid_dir, limit=1600):
    """把 top-k 块原文写盘，供人工核对标准答案。"""
    evid_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {qid} 召回证据", "", f"**问题**：{q}", "",
             f"共 {len(neighbors)} 块。下面按融合排名列出全文（截断 {limit} 字）。", ""]
    for n in neighbors:
        d = c.get(f"/api/chunk?row={n['row']}")
        txt = (d.get("text") or "")[:limit]
        lines += [
            f"## [{n['rank']}] {n['company']} {n['report_year']} · {n['chapter']} · P{n['page']}",
            "",
            f"- chunk_id `{n['chunk_id']}`  row={n['row']}  "
            f"RRF={n['rrf']:.4f} 余弦={n['sim']:.4f}(#{n.get('vrank')}) "
            f"BM25={n.get('bm25', 0):.1f}(#{n.get('brank')})  字数={len(d.get('text') or '')}",
            "",
            "```",
            txt.replace("```", "'''"),
            "```",
            "",
        ]
    p = evid_dir / f"{qid}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def markdown_report(results, meta):
    out = [
        "# 第三周作业 · 逐题评测记录",
        "",
        f"- 语料：{meta.get('scope') or '—'}",
        f"- 索引：{meta.get('n')} 块 / {meta.get('n_pdf')} 份 / dim={meta.get('dim')} / "
        f"{meta.get('model_id')} / 量化={meta.get('quant')}",
        f"- 检索：向量 + BM25，RRF 融合，k={meta.get('k')}",
        f"- 生成：{meta.get('llm_model')}",
        f"- 评测时间：{meta.get('run_at')}",
        "",
        "## 总览",
        "",
        "| 题 | 类型 | 考点 | top-k 判定 | 召回覆盖公司 | map-reduce 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        aggv = (r.get("aggregate") or {}).get("verdict") or "—"
        out.append("| {} | {} | {} | `{}` | {}/16 | {} |".format(
            r["id"], r["kind"], r["aspect"], r["verdict"],
            r.get("n_companies_recalled", "—"), f"`{aggv}`" if aggv != "—" else "—"))
    out.append("")
    out += ["## 逐题明细", ""]
    for r in results:
        out += [
            f"### {r['id']} · {r['aspect']}",
            "",
            f"**问题**：{r['q']}",
            "",
            f"**判定**：`{r['verdict']}` —— {r['reason']}",
            "",
        ]
        if r.get("gold_note"):
            out += [f"**标准答案**：{r['gold_note']}", ""]
        out += [f"**召回 top-{min(12, len(r['neighbors']))}**（融合排名）：", "",
                "| # | 公司 | 章节 | 页 | RRF | 余弦 | 向量名次 | BM25 | BM25名次 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for n in r["neighbors"][:12]:
            out.append("| {} | {} | {} | {} | {:.4f} | {:.4f} | {} | {:.1f} | {} |".format(
                n["rank"], n["company"], n["chapter"][:18], n["page"], n["rrf"],
                n["sim"], n.get("vrank"), n.get("bm25", 0), n.get("brank")))
        out.append("")
        if r.get("answer"):
            out += ["**模型回答**：", "", "> " + r["answer"].replace("\n", "\n> "), ""]
        if r.get("citations"):
            out.append("**引用块**：" + "、".join(
                f"[{c['no']}] {c['company']} {c['chapter'][:10]} P{c['page']}"
                for c in r["citations"][:8]))
            out.append("")
        a = r.get("aggregate")
        if a:
            out += ["**map-reduce（逐家扫描）**：`{}` —— 扫描 {} 家，判定相关 {} 家，"
                    "用时 {:.0f}s".format(a["verdict"], a["n_company"], a["n_relevant"],
                                          a.get("elapsed_s", 0)), ""]
            if a["per_company"]:
                out += ["| 公司 | map 判定 | 抽取到的结论 |", "|---|---|---|"]
                for p in a["per_company"]:
                    out.append("| {} | {} | {} |".format(
                        p["company"], p["state"],
                        (p.get("answer") or "").replace("\n", " ")[:80]))
                out.append("")
            if a.get("answer"):
                out += ["map-reduce 汇总回答：", "", "> " + a["answer"].replace("\n", "\n> "), ""]
        if r.get("note"):
            out += [f"**备注**：{r['note']}", ""]
        out += [f"证据全文：`evidence/{r['id']}.md`", "", "---", ""]
    return "\n".join(out)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8799")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--only", default=None, help="逗号分隔的题目 id")
    ap.add_argument("--no-llm", action="store_true", help="只检索，不调 DeepSeek")
    ap.add_argument("--no-agg", action="store_true",
                    help="跨公司题不跑 map-reduce（只测 top-k 那条路）")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rejudge", nargs="?", const="__default__", default=None,
                    help="只按新的判定规则重新打分已跑完的结果，不调 API（省时间）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    evid_dir = out_dir / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    k = args.k or cfg.get("k", 20)
    cases = cfg["questions"]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in want]

    if args.rejudge:
        src = out_dir / "results.json" if args.rejudge == "__default__" else Path(args.rejudge)
        if not src.exists():
            print(f"找不到 {src}"); return 1
        data = json.loads(src.read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in cfg["questions"]}
        print(f"[重判] {src}")
        for r in data["results"]:
            case = by_id.get(r["id"])
            if not case:
                continue
            old = r["verdict"]
            r["verdict"], r["reason"] = judge(case, r.get("answer") or "")
            print(f"  {r['id']}: {old:10} → {r['verdict']:10}  {r['reason'][:56]}")
            a = r.get("aggregate")
            if a:
                olda = a["verdict"]
                a["verdict"], a["reason"] = judge(case, a.get("answer") or "")
                print(f"  {r['id']} agg: {olda:10} → {a['verdict']:10}  {a['reason'][:56]}")
        src.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "评测记录.md").write_text(
            markdown_report(data["results"], data["meta"]), encoding="utf-8")
        print(f"[完成] 已重写 {src} 与 评测记录.md")
        return 0

    c = Client(args.base)
    h = c.health()
    print(f"[健康] n={h['n']} dim={h['dim']} is_sample={h.get('is_sample')} "
          f"llm_ready={h['llm_ready']}")
    if not h.get("index_ready"):
        print("索引未就绪，先建索引"); return 1
    if not args.no_llm and not h.get("llm_ready"):
        print("DeepSeek 未配置，无法生成答案（可用 --no-llm 只测检索）"); return 1

    results, t_all = [], time.time()
    for i, case in enumerate(cases, 1):
        q = case["q"]
        print("\n" + "=" * 78)
        print(f"【{case['id']}】{case['aspect']}\n{q}")
        t0 = time.time()
        r = c.retrieve(q, k=k)
        nbs = r["neighbors"]
        print(f"  召回 {len(nbs)} 块  embed={r['embed_ms']}ms search={r['search_ms']}ms  "
              f"跨越 {len({n['company'] for n in nbs})} 家公司")
        for n in nbs[:6]:
            print("   [{:>2}] RRF={:.4f} cos={:.4f}(#{}) bm25={:.1f}(#{})  {} {} P{}".format(
                n["rank"], n["rrf"], n["sim"], n.get("vrank"), n.get("bm25", 0),
                n.get("brank"), n["company"], n["chapter"][:14], n["page"]))
        dump_evidence(c, case["id"], q, nbs, evid_dir)

        answer, cites, note = "", [], ""
        if not args.no_llm:
            answer, cites, _ = c.answer(r["qid"], k=len(nbs))
            print(f"  回答 {len(answer)} 字 / 引用 {len(cites)} 条")

        verdict, reason = judge(case, answer)
        print(f"  → {verdict}  {reason}")

        # 跨公司题再跑一遍 map-reduce（逐家扫描）。top-k 天然盖不满 16 家，
        # 两条路都跑才能说清「这类题是检索的问题还是聚合的问题」。
        agg = None
        if case["kind"] == "cross" and not args.no_agg and not args.no_llm:
            print(f"  [map-reduce] 逐家扫描中 …")
            ta = time.time()
            atext, acites, aper, _ = c.aggregate(q, k=6)
            av, ar = judge(case, atext)
            rel = [p for p in aper if p.get("state") == "relevant"]
            print(f"  逐家扫描 {len(aper)} 家 / 判定相关 {len(rel)} 家，用时 {time.time() - ta:.0f}s")
            print(f"  → [map-reduce] {av}  {ar}")
            agg = {
                "answer": atext, "citations": acites, "verdict": av, "reason": ar,
                "n_company": len(aper), "n_relevant": len(rel),
                "per_company": [{"company": p.get("company"), "state": p.get("state"),
                                 "answer": p.get("answer"), "evidence": p.get("evidence")}
                                for p in aper],
                "elapsed_s": round(time.time() - ta, 1),
            }

        results.append({
            "id": case["id"], "kind": case["kind"], "aspect": case["aspect"],
            "q": q, "expect_refusal": bool(case.get("expect_refusal")),
            "gold_note": (case.get("gold") or {}).get("note"),
            "gold_source": (case.get("gold") or {}).get("source"),
            "k": k, "n_neighbors": len(nbs),
            "n_companies_recalled": len({n["company"] for n in nbs}),
            "elapsed_s": round(time.time() - t0, 1),
            "neighbors": nbs,
            "answer": answer,
            "citations": cites,
            "verdict": verdict, "reason": reason, "note": note,
            "aggregate": agg,
        })

    meta = {
        "scope": h.get("scope"), "n": h["n"], "n_pdf": h.get("n_pdf"),
        "dim": h["dim"], "model_id": h.get("model_id"), "quant": h.get("quant"),
        "k": k, "llm_model": h.get("llm_model"),
        "is_sample": h.get("is_sample"),
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "评测记录.md").write_text(markdown_report(results, meta), encoding="utf-8")

    n_ok = sum(1 for r in results if r["verdict"] in ("PASS", "REFUSE_OK"))
    aggs = [r["aggregate"] for r in results if r.get("aggregate")]
    n_agg_ok = sum(1 for a in aggs if a["verdict"] in ("PASS", "REFUSE_OK"))
    print("\n" + "=" * 78)
    print(f"top-k 判定：{n_ok}/{len(results)} 通过  "
          f"（{ {v: sum(1 for r in results if r['verdict'] == v) for v in {r['verdict'] for r in results} } }）")
    if aggs:
        print(f"map-reduce 判定（仅跨公司题）：{n_agg_ok}/{len(aggs)} 通过  "
              f"（{ {v: sum(1 for a in aggs if a['verdict'] == v) for v in {a['verdict'] for a in aggs} } }）")
    print(f"产出：{out_dir / 'results.json'}")
    print(f"      {out_dir / '评测记录.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
