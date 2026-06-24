#!/usr/bin/env python3
"""벤치 결과 JSON → 예쁜 단일 HTML(gpu-bench/index.html) 생성.

사용:
  python gpu-bench/build.py                       # ../results/*.json + gpu-bench/data/*.json 읽음
  python gpu-bench/build.py results/*.json        # 경로 직접 지정
  python gpu-bench/build.py --out gpu-bench/index.html --title "My Bench"

- 데이터를 HTML 안에 임베드 → 파일 그냥 열어도(file://) 표는 보임. 차트는 Chart.js(CDN).
- GitHub Pages 로 gpu-bench/ 를 서빙하면 그대로 웹에 공개됨.
- GPU × 모델 × eager(1/0) 로 묶어서 표 + 라인차트(동시성축) 표시. price 있으면 tok/$ 도.
"""
import argparse
import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_runs(paths):
    runs = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        if "results" not in d:
            continue
        runs.append(d)
    return runs


def organize(runs):
    """{gpu_key: {"meta":..., "models": {model: {eager: [rows...]}}}}"""
    gpus = {}
    for d in runs:
        env = d.get("env", {}) or {}
        gpu = env.get("gpu_name") or "Unknown GPU"
        gc = env.get("gpu_count", 1) or 1
        key = f"{gpu}__x{gc}"
        g = gpus.setdefault(key, {"meta": {
            "gpu": gpu, "gpu_count": gc, "gpu_mem": env.get("gpu_mem"),
            "driver": env.get("driver"), "vllm": env.get("vllm"),
            "torch": env.get("torch"), "price": env.get("price_usd_hr"),
        }, "models": {}})
        # price/메타는 가장 정보 많은 쪽으로 보강
        if env.get("price_usd_hr") and not g["meta"].get("price"):
            g["meta"]["price"] = env.get("price_usd_hr")
        model = d.get("model", "?")
        eager = str(d.get("enforce_eager", "")) or "?"
        rows = d.get("results", [])
        for r in rows:
            r = dict(r)
            price = g["meta"].get("price")
            oa = r.get("output_tok_s_agg")
            r["tok_per_usd"] = round(oa * 3600.0 / float(price)) if (price and oa) else None
        g["models"].setdefault(model, {})[eager] = rows
    return gpus


def eager_label(e):
    return {"1": "eager", "0": "cudagraph"}.get(str(e), f"eager={e}")


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2a3240;--fg:#e6edf3;--mut:#8b949e;
--acc:#3fb950;--acc2:#58a6ff;--warn:#d29922;--rad:14px}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#13283a 0%,var(--bg) 55%);
color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.3px}
.sub{color:var(--mut);margin:0 0 28px;font-size:13px}
.card{background:linear-gradient(180deg,var(--panel) 0%,#13181f 100%);border:1px solid var(--line);
border-radius:var(--rad);padding:20px 22px;margin:0 0 26px;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.gpu-head{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;margin-bottom:6px}
.gpu-name{font-size:19px;font-weight:700}
.badge{font-size:12px;color:var(--mut);background:var(--panel2);border:1px solid var(--line);
border-radius:999px;padding:3px 10px;white-space:nowrap}
.badge b{color:var(--fg);font-weight:600}
.price{margin-left:auto;font-size:18px;font-weight:700;color:#fff;background:#1f6feb;
border-radius:10px;padding:5px 12px}
.model{margin-top:22px}
.model h3{margin:0 0 10px;font-size:15px;color:var(--acc);letter-spacing:.2px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13.5px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--mut);font-weight:600;border-bottom:1px solid var(--line)}
tbody tr:hover{background:#1a2230}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:6px;border:1px solid var(--line)}
.tag.eager{color:var(--warn);border-color:#5a4a1f;background:#241d0c}
.tag.cudagraph{color:var(--acc);border-color:#1f5a2f;background:#0c2414}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:16px}
.chartbox{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 8px 4px}
.chartbox .t{font-size:12px;color:var(--mut);margin:2px 6px 6px}
.note{color:var(--mut);font-size:12px;margin-top:10px}
.kpis{display:flex;gap:18px;flex-wrap:wrap;margin:4px 0 0}
.kpi{font-size:12px;color:var(--mut)}.kpi b{color:var(--fg);font-size:16px;font-weight:700}
footer{color:var(--mut);font-size:12px;margin-top:24px;text-align:center}
a{color:var(--acc2)}
"""


def num(v, d=None):
    return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) and d is not None else str(v))


def render_table(eagers):
    # 컬럼: conc | mode | ttft | tpot | dec | out_agg | req/s | tok/$
    has_price = any(r.get("tok_per_usd") is not None for rows in eagers.values() for r in rows)
    head = ["conc", "mode", "ttft (s)", "tpot (ms)", "decode tok/s", "out tok/s (agg)", "req/s"]
    if has_price:
        head.append("tok/$")
    out = ["<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for eager in sorted(eagers.keys()):
        lab = eager_label(eager)
        for r in sorted(eagers[eager], key=lambda x: x.get("concurrency") or 0):
            cells = [
                num(r.get("concurrency")),
                f'<span class="tag {lab}">{lab}</span>',
                num(r.get("ttft_s_mean")),
                num(r.get("tpot_ms_mean")),
                num(r.get("decode_tok_s_per_req_mean")),
                num(r.get("output_tok_s_agg")),
                num(r.get("req_per_s")),
            ]
            if has_price:
                cells.append(num(r.get("tok_per_usd")))
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def chart_series(eagers, ykey):
    """Chart.js datasets: 한 mode 당 한 라인."""
    colors = {"1": "#d29922", "0": "#3fb950"}
    datasets = []
    xs = sorted({r.get("concurrency") for rows in eagers.values() for r in rows if r.get("concurrency")})
    for eager in sorted(eagers.keys()):
        rows = {r.get("concurrency"): r for r in eagers[eager]}
        data = [rows.get(x, {}).get(ykey) for x in xs]
        datasets.append({
            "label": eager_label(eager),
            "data": data,
            "borderColor": colors.get(str(eager), "#58a6ff"),
            "backgroundColor": colors.get(str(eager), "#58a6ff") + "33",
            "borderWidth": 2, "tension": 0.25, "pointRadius": 3, "spanGaps": True,
        })
    return {"labels": xs, "datasets": datasets}


def build_html(gpus, title):
    chart_specs = []  # (canvas_id, chartjs_config)
    body = []
    cidx = 0
    for gkey in sorted(gpus.keys()):
        g = gpus[gkey]
        m = g["meta"]
        badges = []
        if m.get("gpu_mem"):
            badges.append(f'<span class="badge"><b>VRAM</b> {html.escape(str(m["gpu_mem"]))}</span>')
        if m.get("gpu_count", 1) and m["gpu_count"] != 1:
            badges.append(f'<span class="badge"><b>GPUs</b> {m["gpu_count"]}</span>')
        if m.get("vllm"):
            badges.append(f'<span class="badge"><b>vLLM</b> {html.escape(str(m["vllm"]))}</span>')
        if m.get("torch"):
            badges.append(f'<span class="badge"><b>torch</b> {html.escape(str(m["torch"]))}</span>')
        if m.get("driver"):
            badges.append(f'<span class="badge"><b>driver</b> {html.escape(str(m["driver"]))}</span>')
        price_html = f'<span class="price">${m["price"]}/hr</span>' if m.get("price") else ""

        # headline KPI: 최고 decode tok/s, 최고 throughput (전 모델/모드 통틀어)
        all_rows = [r for mdl in g["models"].values() for rows in mdl.values() for r in rows]
        peak_dec = max([r.get("decode_tok_s_per_req_mean") or 0 for r in all_rows] or [0])
        peak_thr = max([r.get("output_tok_s_agg") or 0 for r in all_rows] or [0])

        body.append('<section class="card">')
        body.append(f'<div class="gpu-head"><span class="gpu-name">{html.escape(m["gpu"])}</span>{"".join(badges)}{price_html}</div>')
        body.append(f'<div class="kpis"><div class="kpi">peak decode <b>{peak_dec:g}</b> tok/s/req</div>'
                    f'<div class="kpi">peak throughput <b>{peak_thr:g}</b> out-tok/s</div></div>')

        for model in sorted(g["models"].keys()):
            eagers = g["models"][model]
            body.append(f'<div class="model"><h3>{html.escape(model)}</h3>')
            body.append(render_table(eagers))
            # charts
            metrics = [("decode_tok_s_per_req_mean", "decode tok/s/req ↑"),
                       ("ttft_s_mean", "TTFT (s) ↓"),
                       ("output_tok_s_agg", "throughput out-tok/s ↑")]
            body.append('<div class="charts">')
            for ykey, tlabel in metrics:
                cidx += 1
                cid = f"c{cidx}"
                cfg = {
                    "type": "line",
                    "data": chart_series(eagers, ykey),
                    "options": {
                        "responsive": True, "maintainAspectRatio": True, "aspectRatio": 1.7,
                        "plugins": {"legend": {"labels": {"color": "#8b949e", "boxWidth": 12, "font": {"size": 11}}}},
                        "scales": {
                            "x": {"title": {"display": True, "text": "concurrency", "color": "#8b949e"},
                                  "ticks": {"color": "#8b949e"}, "grid": {"color": "#2a3240"}},
                            "y": {"beginAtZero": True, "ticks": {"color": "#8b949e"}, "grid": {"color": "#2a3240"}},
                        },
                    },
                }
                chart_specs.append((cid, cfg))
                body.append(f'<div class="chartbox"><div class="t">{html.escape(tlabel)}</div><canvas id="{cid}"></canvas></div>')
            body.append('</div>')  # charts
            body.append('</div>')  # model
        body.append('</section>')

    charts_js = "\n".join(
        f"new Chart(document.getElementById('{cid}'),{json.dumps(cfg)});"
        for cid, cfg in chart_specs)

    note = ('데이터 출처: benchmark.py 결과 JSON. eager=프로덕션 동일(CUDA graph off), '
            'cudagraph=ENFORCE_EAGER=0(best-case decode). TTFT는 ↓낮을수록, 나머지는 ↑높을수록 좋음. '
            'tok/$ = 시간당 가격 대비 출력토큰 처리량(가성비).')

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">Gemma 26B(A4B MoE) / 31B · vLLM 0.20.2 · full-context({{}}) 속도 — GPU별 비교</p>
{''.join(body)}
<p class="note">{html.escape(note)}</p>
<footer>generated by <code>gpu-bench/build.py</code> · gemma-vllm-speedtest</footer>
</div>
<script>{charts_js}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="결과 JSON 경로/glob (생략 시 results/ + gpu-bench/data/)")
    ap.add_argument("--out", default=os.path.join(HERE, "index.html"))
    ap.add_argument("--title", default="Gemma vLLM — GPU Benchmarks")
    args = ap.parse_args()

    paths = []
    srcs = args.paths if args.paths else [os.path.join(ROOT, "results", "*.json"),
                                          os.path.join(HERE, "data", "*.json")]
    for s in srcs:
        paths.extend(glob.glob(s) if any(c in s for c in "*?[") else [s])
    paths = [p for p in paths if os.path.basename(p) != "index.html"]
    if not paths:
        print("결과 JSON 없음 (results/ 나 gpu-bench/data/ 에 넣거나 경로 지정)", file=sys.stderr)
        sys.exit(1)

    runs = load_runs(paths)
    if not runs:
        print("유효한 결과 없음", file=sys.stderr)
        sys.exit(1)
    gpus = organize(runs)
    html_out = build_html(gpus, args.title).replace("full-context({})", "full-context(32k)")
    with open(args.out, "w") as f:
        f.write(html_out)
    n_gpu = len(gpus)
    n_run = len(runs)
    print(f"OK → {args.out}  ({n_gpu} GPU, {n_run} run files)")


if __name__ == "__main__":
    main()
