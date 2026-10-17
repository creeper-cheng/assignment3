# -*- coding: utf-8 -*-
r"""阶段4：用本机 Qwen3-Embedding-0.6B 把 chunks 向量化。

⚠ 必须用 qwen3-embedding/.venv 里的 python 跑（torch 只装在那儿）：
    qwen3-embedding\.venv\Scripts\python.exe rag\embed.py

要点：
  * 默认 **fp32，别改回 int8**（见下面 QUANT 处的实测：动态量化后余弦只剩 0.637）。
  * 两级断点续传：
      份间——vectors/{pid}.npy + .done.json，done 里记 chunks 的 sha256，
            切块参数一改 hash 就变，向量自动作废重算，绝不会向量与元数据错位；
      份内——vectors/{pid}.part.npy + .part.json，每 EMB_CKPT 块 flush 一次。
            一份大年报要跑半小时，整个进程被内存回收器或关机打断时，
            只丢最后一次 checkpoint 之后的几分钟，重跑自动接着做。
  * 写盘顺序：先 npy 再 done.json —— 保证「有 done 必有完整 npy」。
    .part 不带 .done.json，所以 merge_vectors 的 glob 不会捡到半成品。
  * EMB_THREADS 可覆盖线程数（并行跑多个进程分摊语料时用）。

用法:
    python embed.py --sample            # 5 家样本验收（约 4600 块 / 1.7h）
    python embed.py                     # 全量（约 2.7 万块 / 10h）
    python embed.py --compare-quant     # int8 vs fp32 数值差异自检（几十秒）
    python embed.py --quant fp32        # 不用量化
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# 让 import qwen3_embed 落到现有的那个模块（它自己会设 HF_HOME / hf-mirror / 离线开关）
sys.path.insert(0, str(REPO / "qwen3-embedding"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

BATCH = int(os.environ.get("EMB_BATCH", "16"))
# 留 EMB_THREADS 覆盖口：想并行跑多个 embed 进程分摊语料时，
# 让每个进程少占几个核，总线程数贴着物理核数才不会互相抢。
# （实测：两个 16 线程进程抢 16 核，合起来只比单进程快 1.3 倍；
#   16 线程单进程本身也只用满约 2 个核，说明这活儿不是 CPU 密集，
#   加进程收益有限——但至少别让线程数翻倍去自我踩踏。）
THREADS = int(os.environ.get("EMB_THREADS", "16"))
# 份内断点步长：每编码这么多块就 flush 一次 .part.npy。
# 128 块约 7 分钟，被打断最多损失 7 分钟，重跑接着做。
CKPT_EVERY = int(os.environ.get("EMB_CKPT", "128"))

# ⚠ 默认 fp32，别改回 int8。实测（2026-10-02）Qwen3-Embedding-0.6B 在
# torch.ao.quantization.quantize_dynamic 下会严重失真：
#     全量化  vs fp32 平均余弦 = 0.637
#     只量化 attention      = 0.961   （可以接受）
#     只量化 FFN(mlp)       = 0.637   ← 元凶，FFN 占参数大头且有激活离群值
# 0.637 等于换了个模型，检索会完全跑偏。想要 CPU 加速请走 ONNX Runtime 静态量化，
# 不要用 PyTorch 动态量化。
QUANT = "fp32"

# 向量模型可切（EMB_MODEL=bge-small-zh | qwen3-0.6b）。dim / max_len 从 MODEL_SPECS 取，
# 不再写死 1024——写死会让换模型后旧向量被当成合法的继续用。
from qwen3_embed import MODEL_SPECS, current_model_key  # noqa: E402

MODEL_KEY = current_model_key()
MODEL_ID = MODEL_SPECS[MODEL_KEY]["model_id"]
DIM = int(MODEL_SPECS[MODEL_KEY]["dim"])
MAX_LEN = int(MODEL_SPECS[MODEL_KEY]["max_len"])

# 5 家样本：覆盖 TOC 为空+大财务表(000703)、扫描件 OCR(000420)、
# 干净十节基线(002064)、八节版(000949)、标题分行(601233)
SAMPLE_CODES = ["000703", "000420", "002064", "000949", "601233"]

DERIVED = REPO / "年报_合成纤维制造" / "_derived"
CHUNK_DIR = DERIVED / "chunks"
VECTOR_DIR = DERIVED / "vectors"
MANIFEST = DERIVED / "manifest.jsonl"
# 注意：这里不 import ann_common —— 它依赖 pymupdf，而本脚本跑在只有 torch 的
# qwen3-embedding/.venv 里。向量化与抽取层必须解耦。


def setup_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_chunks(pid):
    with open(CHUNK_DIR / f"{pid}.jsonl", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def chunk_digest(chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_sha1"].encode("ascii"))
    return h.hexdigest()


def build_embedder(quant):
    import torch
    from qwen3_embed import make_embedder

    torch.set_num_threads(THREADS)
    emb, _spec = make_embedder(dtype=torch.float32, verbose=True)
    if quant == "int8":
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            from torch.quantization import quantize_dynamic
        t0 = time.time()
        emb.model = quantize_dynamic(emb.model, {torch.nn.Linear}, dtype=torch.qint8)
        print(f"[quant] int8 动态量化完成 {time.time() - t0:.1f}s")
    return emb


def done_ok(pid, digest, quant):
    d = VECTOR_DIR / f"{pid}.done.json"
    n = VECTOR_DIR / f"{pid}.npy"
    if not (d.exists() and n.exists()):
        return None
    try:
        rec = json.load(open(d, encoding="utf-8"))
    except Exception:
        return None
    if (rec.get("chunks_sha256") == digest and rec.get("quant") == quant
            and rec.get("model_id") == MODEL_ID and rec.get("dim") == DIM):
        return rec
    return None


def compare_quant(n=20):
    """int8 vs fp32 数值差异自检——量化不该带来 >1e-2 的余弦偏移。"""
    import numpy as np
    import torch

    files = sorted(CHUNK_DIR.glob("*.jsonl"))
    texts = []
    for f in files:
        for c in load_chunks(f.stem):
            texts.append(c["embed_text"])
            if len(texts) >= n:
                break
        if len(texts) >= n:
            break
    if not texts:
        print("没有 chunks")
        return 1

    res = {}
    for q in ("fp32", "int8"):
        e = build_embedder(q)
        t0 = time.time()
        v = e.encode(texts, batch_size=BATCH, max_length=MAX_LEN).numpy()
        dt = time.time() - t0
        res[q] = v
        print(f"[{q}] dim={v.shape[1]} norm={np.linalg.norm(v, axis=1).mean():.4f} "
              f"耗时 {dt:.1f}s ({dt / len(texts) * 1000:.0f} ms/块)")
        del e

    cos = (res["fp32"] * res["int8"]).sum(axis=1)
    diff = 1 - cos
    print(f"[对比] 余弦相似度 fp32↔int8: min={cos.min():.5f} mean={cos.mean():.5f}")
    print(f"[对比] 最大 1-cos = {diff.max():.5f}  ({'✓ 通过' if diff.max() < 1e-2 else '✗ 超过 1e-2'})")
    return 0 if diff.max() < 1e-2 else 1


def main(argv=None):
    setup_console()
    ap = argparse.ArgumentParser(description="年报块向量化")
    ap.add_argument("--only", help="指定 pdf_id，逗号分隔")
    ap.add_argument("--year", type=int, action="append",
                    help="只跑指定报告期（可重复，如 --year 2025）。"
                         "比 --only 安全：不用在命令行传中文 pdf_id，避免 Windows GBK 控制台弄坏参数")
    ap.add_argument("--corpus", default=None,
                    help="按 rag/corpus_2025.json 声明的公司清单+报告期过滤。"
                         "build_index.bat 用它，免得一默认就把全部 92 份都向量化了")
    ap.add_argument("--existing", action="store_true",
                    help="只处理已有向量产物的那批（切块变更后重算用）。"
                         "比 --only 安全：不用在命令行传中文 pdf_id")
    ap.add_argument("--sample", action="store_true", help=f"只跑 5 家样本 {SAMPLE_CODES}")
    ap.add_argument("--quant", default=QUANT, choices=["int8", "fp32"])
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compare-quant", action="store_true", help="只做 int8/fp32 数值自检")
    args = ap.parse_args(argv)

    if args.compare_quant:
        return compare_quant()

    import numpy as np

    with open(MANIFEST, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    if args.sample:
        recs = [r for r in recs if r["stock_code"] in SAMPLE_CODES]
    if args.corpus:
        cp = Path(args.corpus)
        if not cp.is_absolute():
            cp = REPO / cp
        ccfg = json.load(open(cp, encoding="utf-8"))
        codes = {c["code"] for c in ccfg.get("companies", [])}
        cyear = int(ccfg.get("report_year") or 0)
        before = len(recs)
        recs = [r for r in recs
                if (not codes or r["stock_code"] in codes)
                and (not cyear or r["report_year"] == cyear)]
        print(f"[语料] {cp.name}：{ccfg.get('scope', '')} → 选中 {len(recs)}/{before} 份")
    if args.year:
        want_years = set(args.year)
        recs = [r for r in recs if r["report_year"] in want_years]
    if args.existing:
        have = {p.name[:-len(".done.json")] for p in VECTOR_DIR.glob("*.done.json")}
        recs = [r for r in recs if r["pdf_id"] in have]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        recs = [r for r in recs if r["pdf_id"] in want or r["stock_code"] in want]
    # 只处理已经切好块的
    recs = [r for r in recs if (CHUNK_DIR / f"{r['pdf_id']}.jsonl").exists()]
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        print("没有可处理的 chunks")
        return 1

    todo, skip = [], 0
    for r in recs:
        chunks = load_chunks(r["pdf_id"])
        dg = chunk_digest(chunks)
        if not args.force and done_ok(r["pdf_id"], dg, args.quant):
            skip += 1
            continue
        todo.append((r, chunks, dg))

    total_chunks = sum(len(c) for _, c, _ in todo)
    print("=" * 70)
    print(f"向量化 {len(todo)} 份 / {total_chunks} 块  （已完成跳过 {skip} 份）")
    print(f"模型 {MODEL_ID}（{MODEL_KEY}，dim={DIM}）  量化={args.quant}  batch={args.batch}  threads={THREADS}")
    est = total_chunks * (1.3 if args.quant == "int8" else 2.3)
    print(f"预计耗时 ~{est / 3600:.1f} 小时（按旧模型的单块耗时估，仅供参考）")
    print("=" * 70)
    if not todo:
        return 0

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    emb = build_embedder(args.quant)
    t_start = time.time()
    done_chunks = 0

    for n, (rec, chunks, dg) in enumerate(todo, 1):
        pid = rec["pdf_id"]
        texts = [c["embed_text"] for c in chunks]
        nch = len(chunks)
        npy = VECTOR_DIR / f"{pid}.npy"
        # ⚠ 份内断点：一件 PDF 要跑半小时，进程被内存回收器/关机打断时，
        # 原来「跑完一整份才落盘」等于整份白干。改成边跑边写 .part.npy，
        # 元数据记到第几块；跑完再把 .part.npy 直接改名成正式 npy。
        # .part 没有配套 .done.json，所以 merge_vectors 的 glob 不会捡到半成品。
        part = VECTOR_DIR / f"{pid}.part.npy"
        pmeta = VECTOR_DIR / f"{pid}.part.json"

        start = 0
        if part.exists() and pmeta.exists() and not args.force:
            try:
                pm = json.load(open(pmeta, encoding="utf-8"))
                if (pm.get("chunks_sha256") == dg and pm.get("quant") == args.quant
                        and pm.get("model_id") == MODEL_ID and pm.get("dim") == DIM
                        and int(pm.get("total", -1)) == nch):
                    start = max(0, min(int(pm.get("done", 0)), nch))
            except Exception:
                start = 0
        try:
            if start and part.exists():
                buf = np.lib.format.open_memmap(part, mode="r+", dtype=np.float32,
                                                shape=(nch, DIM))
            else:
                start = 0
                buf = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32,
                                                shape=(nch, DIM))
        except Exception:
            start = 0
            buf = np.lib.format.open_memmap(part, mode="w+", dtype=np.float32,
                                            shape=(nch, DIM))
        if start:
            print(f"[{n}/{len(todo)}] {pid} 续跑：已有 {start}/{nch} 块")

        failed = None
        try:
            while start < nch:
                end = min(nch, start + CKPT_EVERY)
                v = emb.encode(texts[start:end], batch_size=args.batch,
                               max_length=MAX_LEN).numpy()
                if v.shape[1] != DIM:
                    failed = f"维度异常 {v.shape}"
                    break
                buf[start:end] = v.astype("float32")
                start = end
                buf.flush()
                json.dump({"chunks_sha256": dg, "model_id": MODEL_ID, "quant": args.quant,
                           "dim": DIM, "total": nch, "done": start},
                          open(pmeta, "w", encoding="utf-8"), ensure_ascii=False)
        except KeyboardInterrupt:
            print(f"\n[中断] {pid} 已做到 {start}/{nch} 块，进度在 .part，重跑接着做。")
            raise
        except Exception as e:
            failed = f"{type(e).__name__}: {e}"

        if failed:
            print(f"[{n}/{len(todo)}] {pid} ✗ {failed}（已做 {start}/{nch} 块，保留在 .part）")
            continue

        del buf
        os.replace(part, npy)                     # .part 就是完整矩阵，先落 npy
        v = np.load(npy, mmap_mode="r")
        h = hashlib.sha256()
        for i in range(0, v.shape[0], 256):
            h.update(np.ascontiguousarray(v[i:i + 256]).tobytes())
        meta = {
            "chunks_sha256": dg, "model_id": MODEL_ID, "quant": args.quant,
            "max_length": MAX_LEN, "dim": int(v.shape[1]),
            "count": int(v.shape[0]), "dtype": "float32",
            "embed_sha256": h.hexdigest(),
        }
        with open(VECTOR_DIR / f"{pid}.done.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)   # 后 done.json
        pmeta.unlink(missing_ok=True)

        done_chunks += len(chunks)
        el = time.time() - t_start
        rate = done_chunks / el if el else 0
        left = (total_chunks - done_chunks) / rate if rate else 0
        print(f"[{n}/{len(todo)}] {pid} {len(chunks)}块 ✓  "
              f"{rate:.2f}块/s  已用{el / 60:.1f}分  剩余~{left / 3600:.1f}小时")

    print("-" * 70)
    print(f"完成 {done_chunks} 块，用时 {(time.time() - t_start) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
