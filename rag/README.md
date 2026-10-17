# 年报 RAG · 过程可视化

把 A 股年报做成可检索的向量库，并把 RAG 的**内部过程**画出来：
散点图看聚类、点开看原文、提问后分段演示「问题 → 向量 → 找邻居 → 变答案」，
最后交 DeepSeek 生成带引用的回答。

## 它长什么样

- **散点图**：全部文本块的向量降到二维，按章节着色，一眼看出聚成哪几团
- **点任意一点** → 展开该块原文（公司 / 年份 / 章节 / 页码）
- **单文档问答**：提问后依次显示 ① 向量前 8 个真实数值 → ② 向量落在散点图的位置
  → ③ 逐条画出召回的邻居与相似度 → ④ DeepSeek 流式生成带 `[n]` 引用的回答
- **跨文档聚合**：逐家扫描全部年报、各抽一份结构化结论，再汇总回答问题
  （「哪些公司计提了存货跌价准备，它们之后的产能计划是什么」这类问题，
  top-k 检索天然答不了，必须 map-reduce）

## 本次语料（第三周作业）

合成纤维制造行业 **16 家 × 2025 年年度报告**，共 **7,377 块**，单一年度口径可比。
选择清单与理由见 `rag/corpus_2025.json`。全量是 32 家 / 92 份 / 39,996 块，
取 16 家一年是作业要求（「10 家以上」）与算力之间的折中。

### 向量模型：bge-small-zh-v1.5（默认），Qwen3-0.6B 可切

本机只有 CPU，两个模型的实测差距是**数量级**的：

| 模型 | 参数 | 维度 | 实测吞吐 | 7,377 块耗时 |
|---|---|---|---|---|
| `BAAI/bge-small-zh-v1.5` | 24M | 512 | **11.8 块/s** | **~13 分钟** |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | 0.29 块/s | ~7 小时 |

一开始用的是 Qwen3-0.6B，跑到第一份年报就被系统内存回收器打断了两次（进程常驻 ~5.9 GB，
在 16 GB 机器上被判为内存紧张），于是换成 bge-small-zh。**代价是检索质量低一档**，
在「跨章节勾稽」「定性召回」这类题上能明显看出（见 `rag/eval/结论.md`）。

切换用 `EMB_MODEL`：`bge-small-zh`（默认）或 `qwen3-0.6b`。
`dim` / `max_len` 从 `qwen3_embed.py` 的 `MODEL_SPECS` 取，不再写死 1024——
写死的话换模型后旧向量会被当成合法的继续用，是最隐蔽的一类错位。
换模型必须**整库重算**：`merge_vectors.py` 会校验同一批向量模型一致，混了就拒绝建索引。

索引是否「覆盖了宣称的那份语料」，由 `build_viz_index.py` 跟 `corpus_2025.json`
对**集合**判定（不是靠块数阈值猜），结果写进 `build_meta.json` 的 `is_sample` / `scope`，
前端顶部横幅直接显示。补齐年份或公司后重跑 `build_index.bat` 即可，代码不用改。

## 评测（第三周作业）

`rag/eval/` 是「逐题记录召回了哪些块、答对没有、错在哪」这条要求的落地：

| 文件 | 作用 |
|---|---|
| `questions.json` | 10 道题（2 表内数字 + 1 跨章节 + 1 扭亏 + 1 单公司定性 + 4 跨公司 + 1 越界），每题写明标准答案与出处块 |
| `run_eval.py` | 跑完整链路，落盘 `out/results.json` + `out/评测记录.md` + `out/evidence/*.md` |
| `gold_from_source.py` | **直接从切块原文取标准答案**，不经检索、不经大模型 |
| `sweep.py` | 跨公司题逐家扫描（走检索），用来交叉复核 `gold_from_source` 的结论 |
| `shot.py` | Playwright 驱动本机 Edge 拍页面截图 |

**标准答案不能拿 RAG 自己的输出当**，那是循环论证。所以 `gold_from_source.py`
只做一件事：按字段名把含该字段的原文块捞出来给人看，人不做推断、只读原文。
跨公司题再逐家核对——比如「哪两家亏损」，是逐家把第2节归母净利润读出来比出来的；
中复神鹰那张表的 OCR 列序是错位的，用分季度合计（9,618.41 万元）和 EPS 0.11
反推才确认 2025 年是 96,184,096.91。

判定规则：正向题看回答里有没有 `gold.must` 的关键数字（归一化后比对，容忍千分位/空格差异，
也接受「20.33 亿」这种按亿/万改写）；负向题反过来——**出现大额数字即判错**，
说明它拿别的年份的数字冒充或者干脆编了一个。

跨公司题会**同时**跑 top-k 和 map-reduce（逐家扫描）两条路，
因为 top-k 天然盖不满 16 家，两条都跑才能分清「是检索没召回，还是聚合没做好」。

```bash
python rag\eval\run_eval.py                 # 全部 10 题
python rag\eval\run_eval.py --no-llm        # 只测检索不调 DeepSeek
```

## 环境分裂与解法

本机 torch 只装在 `qwen3-embedding/.venv`，jieba 只在主 Python，两边都没有 Web 框架。
所以不把 torch 拖进服务进程，而是**常驻子进程 + 行协议**：

```
主环境 (pythoncore-3.14)                venv (qwen3-embedding/.venv)
  ├─ HTTP 服务      app.py        ⇄      └─ 常驻编码器  embed_worker.py
  ├─ 建索引         build_viz_index.py        （模型加载一次驻留，
  └─ DeepSeek 调用                            stdin/stdout 一行一 JSON）
```

好处：UI 秒开、模型加载不阻塞服务、worker 挂了可单独重启、不动 venv。

## 流水线

| 阶段 | 脚本 | 产出 |
|---|---|---|
| 0 清单 | `build_manifest.py` | `_derived/manifest.jsonl` |
| 1 抽取 | `extract.py` | `_derived/text/{pdf_id}.json` |
| 2 切块 | `chunk.py` | `_derived/chunks/{pdf_id}.jsonl` |
| 3 校验 | `check_chunks.py` | 切块质量断言 |
| 4 向量化 | `embed.py` | `_derived/vectors/{pdf_id}.npy` |
| 5 合并 | `merge_vectors.py` | `vectors.npy` + `index.json` |
| 6 建索引 | `build_viz_index.py` | `viz/{coords,groups,bm25,offsets}` |
| 7 服务 | `app.py` | http://localhost:8765/ |

跑：仓库根目录双击 `build_index.bat` → `run_rag.bat`。

## 几个踩过的坑（都写进代码注释了）

**表格还原不能用 `page.find_tables()`**。它会漏掉无框线财务表、把整段正文误判成表格、
合并单元格输出一堆 `None`。本项目用坐标聚类，并且是**两步**：

1. **先把纵向相邻、横向重叠的词合并成「单元格」**——年报表格的单元格经常折行
   （`3,925,279,749.` / `60`）或纵向跨多行（`归属于上市公司` / `股东的净利润` / `（元）`）
2. **再按纵向重叠把单元格聚成「行」**
3. 收尾把"标签折到第三行"导致的分家再拼回去

原来只做「按视觉行聚行 + 行内切列」，在行与行 y 不对齐的表上会散架——实测某页
7 列 + 二级表头的「主要会计数据」表，标签列被分到别的行、表头被切碎，
导致「归属于上市公司股东的净利润」这一行彻底对不上数字。

**页眉页脚必须「位置带内统计频次」**。某些高频短语（`√适用□不适用` 出现 146 次、
`单位：元币种：人民币` 112 次）远超常规频次阈值，但都是有语义的正文。
所以频次只在页面顶部/底部 12% 的位置带内统计。

**向量不能用 PyTorch 动态 int8 量化**。实测 Qwen3-Embedding-0.6B 量化后与 fp32
的平均余弦只剩 **0.637**（元凶是 FFN），等于换了个模型，**而且不报错、静默变差**。
索引与查询两侧都必须 fp32。

**召回必须混合（向量 + BM25，RRF 融合）**。纯向量对"某个数字是多少"这类查询很弱：
实测答案块向量排 404 名、BM25 排第 2 名。RRF 只看名次不看分值，
避免余弦(0~1) 和 BM25(0~30) 量纲不同互相压制。

**k 要够大**。「经营活动现金流与净利润差异」这类问题，需要的「现金流量表补充资料」
融合后排 15–24 名，`k=8` 直接漏掉，模型只能从别处抓一个口径不对的净利润。
默认 `k=20`。

**切块与向量用内容哈希绑定**。`embed.py` 把各块 `chunk_sha1` 的 sha256 写进
`done.json`，`merge_vectors.py` 会比对——切块一变，旧向量自动作废而不是静默错位。

**Windows 批处理必须纯 ASCII + CRLF**。cmd 按字节偏移分块读批处理文件，
`chcp 65001` 一改代码页，字节到字符的映射就错位，中文行会被从中间截断当成新命令执行
（症状：窗口一闪就没）。所以 `.bat` 全英文，中文输出交给 Python。

## 目录

```
rag/
  ann_common.py        路径、页质量判定/OCR、页眉页脚剔除、表格还原
  build_manifest.py    索引.csv → manifest.jsonl
  extract.py           逐页抽取文字 + 表格行列还原（坏页回退 OCR）
  chunk.py             章节识别 + 800 字切块（重合 120，句末收尾）
  check_chunks.py      切块质量断言
  embed.py             向量化（断点续传，内容哈希防错位）
  merge_vectors.py     合并成 vectors.npy + index.json
  search_cli.py        命令行检索
  build_viz_index.py   UMAP 降维 + BM25 倒排 + 字节偏移表
  app.py               HTTP 服务 + 全部接口 + DeepSeek 流式
  embed_worker.py      venv 侧常驻编码器
  web/index.html       单文件前端（原生 JS，零 CDN）
  ragcli.py            检索/问答的 HTTP 客户端（评测脚本共用）
  corpus_2025.json     本次语料的公司清单与选择理由
  _bench_embed.py      吞吐基准（0.29 块/秒，选范围时用）
  eval/                questions.json · run_eval.py · sweep.py · shot.py · out/
  tests/               smoke.py（18 项）· acceptance.py（问答验收）
```

数据与模型不入库：`年报_合成纤维制造/`（PDF + 切块 + 向量）、
`qwen3-embedding/`（venv + 1.2 GB 模型权重）、`*.env`（密钥）。
