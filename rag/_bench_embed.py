# -*- coding: utf-8 -*-
"""一次性基准：测量本机 Qwen3-Embedding-0.6B 的真实吞吐（块/秒）。

必须用 venv 跑：
    qwen3-embedding\.venv\Scripts\python.exe rag\_bench_embed.py
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "qwen3-embedding"))

CHUNK_DIR = REPO / "年报_合成纤维制造" / "_derived" / "chunks"

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    import torch
    from qwen3_embed import QwenEmbedder

    n_threads = int(os.environ.get("EMB_THREADS", "16"))
    torch.set_num_threads(n_threads)
    print(f"torch {torch.__version__}  threads={n_threads}  "
          f"cuda={torch.cuda.is_available()}  interop={torch.get_num_interop_threads()}")

    t0 = time.time()
    emb = QwenEmbedder(dtype=torch.float32, verbose=False)
    print(f"模型加载 {time.time() - t0:.1f}s")

    # 取真实块，尽量模拟全量分布
    texts = []
    for f in sorted(CHUNK_DIR.glob("*.jsonl")):
        for c in open(f, encoding="utf-8"):
            if c.strip():
                texts.append(json.loads(c)["embed_text"])
        if len(texts) >= 64:
            break
    texts = texts[:64]
    lens = [len(t) for t in texts]
    print(f"样本 {len(texts)} 块, 平均 {sum(lens) / len(lens):.0f} 字, "
          f"最长 {max(lens)} 字")

    emb.encode(["预热"], is_query=True)   # 吃掉首次内核选择

    for bs in (16, 32):
        t0 = time.time()
        emb.encode(texts, batch_size=bs, max_length=1024)
        dt = time.time() - t0
        print(f"batch={bs:<3} {dt:.2f}s  {len(texts) / dt:.2f} 块/s  "
              f"{dt / len(texts) * 1000:.0f} ms/块")
    return 0


if __name__ == "__main__":
    sys.exit(main())
