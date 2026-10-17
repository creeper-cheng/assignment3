# 第三周作业 · 年报问答知识库

把 A 股年报做成可检索的知识库：抽取文字与表格 → 带出处切块 → 向量 + BM25 混合索引 →
能提问、答案带引用，并把 RAG 的内部过程画在页面上。

**语料**：合成纤维制造行业 **16 家上市公司 × 2025 年年度报告**，**7,377 块**。
**索引**：BAAI/bge-small-zh-v1.5（512 维，fp32）+ BM25，RRF 融合，k=20。
**生成**：DeepSeek，答案带 `[n]` 引用块。

## 交付物

| 内容 | 位置 |
|---|---|
| 代码仓库 | `rag/`（抽取 → 切块 → 向量 → 索引 → 问答服务） |
| 页面截图 | `截图/01_全景散点.png`、`02_检索过程.png`、`03_回答与引用.png`、`04_问答面板.png` |
| 10 道题 + 一页结论 | `第三周作业_题目与结论.pdf` |
| 逐题评测记录 | `rag/eval/out/评测记录.md`、`results.json`、`evidence/qNN.md` |
| 结论（markdown 源） | `rag/eval/结论.md` |

## 怎么跑

```bat
rem 1. 配密钥（本仓库不含密钥）
rem    新建 deepseek.env，三行：
rem    LLM_BASE_URL=https://api.deepseek.com
rem    LLM_API_KEY=sk-你的key
rem    LLM_MODEL=deepseek-chat

rem 2. 下载年报（约 580 MB，不入库）
python download_annual_reports.py

rem 3. 抽取 + 切块
python rag\extract.py
python rag\chunk.py

rem 4. 建索引（16 家约 13 分钟）
build_index.bat

rem 5. 起页面 → http://localhost:8765/
run_rag.bat
```

评测与截图（页面开着时）：

```bat
python rag\eval\run_eval.py        rem 跑 10 道题，含 4 道跨公司题的逐家扫描
python rag\eval\make_pdf.py        rem 重新生成题目+结论 PDF
python rag\eval\shot.py --out 截图 rem 重新截图
```

## 结论摘要

- **单公司 + 表内唯一数字**（q01/q03/q04）最稳，数字与原文逐位一致。
- **跨章节勾稽会抓错口径**（q02）：经营现金流对了，净利润抓成了「综合收益总额」，答对一半、错一半，且错得很像对的。
- **跨公司题 top-k 结构性覆盖不足**（20 块平摊到 16 家），必须靠 map-reduce 逐家扫描；但 map 能否成功取决于**是否按实体提问**——q09 成、q08 败，差别就在这句提问怎么写。
- **拒答好过错答**：越界题（问 2022 年）正确说明语料只有 2025 年，没有拿 2023 列冒充。

完整判读见 `rag/eval/结论.md` 或 PDF 最后一节。**自动判定会系统性高估**（关键词匹配分不清
「X 说了 Y」和「X 未提及 Y」），表上的 4/10 已是修正后的数，仍需抽人看原文。

## 数据不入库

原始 PDF、切块、向量、模型权重合计约 2.8 GB，未提交（见 `.gitignore`）。
按上面的步骤可以完全重建；索引产物与切块内容通过 `chunk_sha1` 绑定，
切块一变旧向量自动作废，不会静默错位。

## 目录

```
rag/                 流水线（详见 rag/README.md）
  ann_common.py      路径、页质量判定/OCR、页眉页脚剔除、表格还原
  extract.py         逐页抽取文字 + 表格行列还原
  chunk.py           章节识别 + 800 字切块（带公司/章节/页码）
  embed.py           向量化（两级断点续传）
  merge_vectors.py   合并成 vectors.npy + index.json
  build_viz_index.py UMAP 降维 + BM25 倒排 + 字节偏移表
  app.py             HTTP 服务 + 全部接口 + DeepSeek 流式
  web/index.html     单文件前端（原生 JS，零 CDN）
  eval/              评测：questions.json · run_eval.py · 结论.md
截图/                页面截图
qwen3-embedding/     向量模型封装（模型权重不入库）
```
