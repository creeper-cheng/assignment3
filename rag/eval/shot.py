# -*- coding: utf-8 -*-
r"""给交付物拍页面截图。

用 Playwright 驱动本机已装的 Edge（channel=msedge），不额外下载浏览器。
抓三张：全景散点 → 检索过程 → 回答与引用。

用法（先把 app.py 跑起来）：
    python rag\eval\shot.py --base http://127.0.0.1:8799 --out 截图
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

DEFAULT_Q = "在这16家合成纤维行业公司中，2025年归属于上市公司股东的净利润为负（亏损）的有哪几家？"


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
    ap.add_argument("--out", default=str(REPO / "截图"))
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=2.0, help="deviceScaleFactor，2 更清晰")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(**kw)
                print(f"[浏览器] launch({kw}) 成功")
                break
            except Exception as e:
                print(f"[浏览器] launch({kw}) 失败：{str(e)[:90]}")
        if browser is None:
            print("没有可用的浏览器（Edge/Chrome/Playwright chromium 都起不来）")
            return 1

        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=args.scale,
        )
        page = ctx.new_page()
        page.on("console", lambda m: print(f"  [console.{m.type}] {m.text[:160]}")
                if m.type == "error" else None)

        print(f"[打开] {args.base}/")
        page.goto(args.base + "/", wait_until="domcontentloaded", timeout=60_000)

        # 等散点画完 + 向量模型就绪（页面自己在轮询 /api/health）
        print("[等待] 索引载入 + 模型就绪 ...")
        page.wait_for_function(
            "() => { const s=document.getElementById('stat');"
            " return s && !s.textContent.includes('载入中')"
            " && !document.getElementById('go').disabled; }",
            timeout=180_000)
        page.wait_for_timeout(2500)          # 让 canvas 动画/字体稳定

        f1 = out / "01_全景散点.png"
        page.screenshot(path=str(f1))
        print(f"[截图] {f1}")

        # ---- 提问 ----
        print(f"[提问] {args.question}")
        page.fill("#q", args.question)
        page.click("#go")

        # 第 3 张卡片（邻居列表）出现后，滚动到检索过程
        try:
            page.wait_for_selector("#hits .hit", timeout=180_000)
        except Exception as e:
            print(f"[警告] 等邻居列表超时：{str(e)[:100]}")
        page.wait_for_timeout(1200)
        page.eval_on_selector("#steps", "el => el.scrollTop = 0")
        page.wait_for_timeout(400)
        f2 = out / "02_检索过程.png"
        page.screenshot(path=str(f2))
        print(f"[截图] {f2}")

        # ---- 等回答流式结束 ----
        print("[等待] DeepSeek 生成 ...")
        try:
            page.wait_for_function(
                "() => { const a=document.getElementById('ans');"
                " const c=document.getElementById('chips');"
                " return a && a.textContent.trim().length>20 && c && c.children.length>0; }",
                timeout=300_000)
        except Exception as e:
            print(f"[警告] 等回答超时：{str(e)[:100]}")
        page.wait_for_timeout(1500)
        # 直接让「回答卡片」自己滚进视口。设 #steps.scrollTop=scrollHeight 不可靠：
        # 回答流式追加期间 scrollHeight 一直在变，滚完又被顶回去。
        for _ in range(3):
            try:
                page.eval_on_selector(
                    "#ans",
                    "el => el.closest('.card').scrollIntoView({block:'start'})")
            except Exception as e:
                print(f"[警告] 回答卡片无法滚入视口：{str(e)[:100]}")
                break
            page.wait_for_timeout(500)
        pos = page.evaluate(
            "() => { const s=document.getElementById('steps');"
            " return [s.scrollTop, s.scrollHeight, s.clientHeight]; }")
        print(f"[滚动] steps scrollTop={pos[0]} scrollHeight={pos[1]} clientHeight={pos[2]}")
        page.wait_for_timeout(400)
        f3 = out / "03_回答与引用.png"
        page.screenshot(path=str(f3))
        print(f"[截图] {f3}")

        # 顺手把右侧面板单独存一张，便于放进报告
        try:
            page.locator("#right").screenshot(path=str(out / "04_问答面板.png"))
            print(f"[截图] {out / '04_问答面板.png'}")
        except Exception as e:
            print(f"[警告] 面板截图失败：{str(e)[:100]}")

        ctx.close()
        browser.close()
    print(f"[完成] → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
