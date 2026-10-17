# -*- coding: utf-8 -*-
r"""问答 / 检索的 HTTP 客户端 —— 给验收脚本和评测脚本共用。

不依赖第三方库（requests 在主环境有，但用 urllib 保证任何 python 都能跑）。
    from ragcli import Client
    c = Client("http://127.0.0.1:8799")
    r = c.retrieve("泰和新材2025年营业收入", k=20)
    a, cites = c.answer(r["qid"], k=len(r["neighbors"]))
"""
import json
import urllib.error
import urllib.request


class Client:
    def __init__(self, base="http://127.0.0.1:8799", timeout=600):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _req(self, path, obj=None):
        url = self.base + path
        if obj is None:
            return urllib.request.urlopen(url, timeout=self.timeout)
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def get(self, path):
        with self._req(path) as r:
            return json.loads(r.read().decode("utf-8"))

    def post(self, path, obj):
        with self._req(path, obj) as r:
            return json.loads(r.read().decode("utf-8"))

    def health(self):
        return self.get("/api/health")

    def retrieve(self, query, k=20, mode=None):
        body = {"query": query, "k": k}
        if mode:
            body["mode"] = mode
        return self.post("/api/retrieve", body)

    def answer(self, qid, k, on_token=None):
        """消费 /api/answer 的 NDJSON 流，返回 (answer_text, citations, events)。"""
        return self._stream({"qid": qid, "k": k}, "/api/answer", on_token=on_token)[:3]

    def aggregate(self, query, k=6, companies=None, on_company=None, timeout=1800):
        """跨文档 map-reduce（逐家扫描 → 汇总）。

        返回 (answer_text, citations, per_company, events)。
        per_company 是 map 阶段每家一条：{company, state, answer, evidence}，
        state ∈ relevant / irrelevant / empty / error。逐家扫描比 top-k 慢得多
        （每家一次 LLM 调用），所以超时放宽到 30 分钟。
        """
        old, self.timeout = self.timeout, timeout
        try:
            body = {"query": query, "k": k}
            if companies:
                body["companies"] = list(companies)
            text, cites, events = self._stream(body, "/api/aggregate",
                                               on_token=None, on_company=on_company)
        finally:
            self.timeout = old
        per = [e for e in events if e.get("type") == "company"]
        return text, cites, per, events

    def _stream(self, body, path, on_token=None, on_company=None):
        """消费 NDJSON 流，返回 (text, cites, events)。"""
        resp = self._req(path, body)
        buf, text, cites, events = "", "", [], []
        for raw in resp:
            buf += raw.decode("utf-8")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if not line.strip():
                    continue
                o = json.loads(line)
                events.append(o)
                t = o.get("type")
                if t == "token":
                    text += o["t"]
                    if on_token:
                        on_token(o["t"])
                elif t == "citations":
                    cites = o["items"]
                elif t == "company" and on_company:
                    on_company(o)
        return text, cites, events
