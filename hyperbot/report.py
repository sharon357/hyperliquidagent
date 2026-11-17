"""Self-contained HTML backtest report (inline SVG, no JavaScript, no external files)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _polyline(series: pd.Series, lo: float, hi: float, w: int, h: int, pad: int) -> str:
    n = len(series)
    if n < 2 or hi == lo:
        return ""
    step = max(n // 1500, 1)  # keep the file small
    vals = series.iloc[::step].to_numpy()
    xs = [pad + i * (w - 2 * pad) / (len(vals) - 1) for i in range(len(vals))]
    ys = [h - pad - (v - lo) / (hi - lo) * (h - 2 * pad) for v in vals]
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _chart(lines: list[tuple[str, pd.Series, str]], title: str, w: int = 900, h: int = 280) -> str:
    pad = 34
    lo = min(float(s.min()) for _, s, _ in lines)
    hi = max(float(s.max()) for _, s, _ in lines)
    body = [f'<text x="{pad}" y="18" class="t">{html.escape(title)}</text>']
    body.append(f'<text x="4" y="{pad + 4}" class="a">{hi:,.0f}</text>')
    body.append(f'<text x="4" y="{h - pad + 4}" class="a">{lo:,.0f}</text>')
    legend_x = pad
    for name, s, color in lines:
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{_polyline(s, lo, hi, w, h, pad)}"/>')
        body.append(f'<rect x="{legend_x}" y="{h - 14}" width="10" height="3" fill="{color}"/>')
        body.append(f'<text x="{legend_x + 14}" y="{h - 9}" class="a">{html.escape(name)}</text>')
        legend_x += 30 + 7 * len(name)
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">{"".join(body)}</svg>'


def write_report(result, close: pd.Series, coin: str, interval: str, path: Path) -> Path:
    eq = result.equity
    bh = close.reindex(eq.index) / close.reindex(eq.index).iloc[0] * result.config.initial_equity
    dd = (eq / eq.cummax() - 1) * 100
    rows = "".join(
        f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{v:,}</td></tr>" if isinstance(v, (int, float)) else ""
        for k, v in result.metrics.items()
    )
    trades = result.trades.tail(200)
    trade_rows = "".join(
        f"<tr><td>{t.entry_time:%Y-%m-%d %H:%M}</td><td>{t.side}</td><td>{t.entry_px:,.4g}</td>"
        f"<td>{t.exit_time:%Y-%m-%d %H:%M}</td><td>{t.exit_px:,.4g}</td>"
        f"<td class=\"{'pos' if t.pnl > 0 else 'neg'}\">{t.return_pct:+.2f}%</td><td>{html.escape(str(t.reason))}</td></tr>"
        for t in trades.itertuples()
    )
    params = ", ".join(f"{k}={v}" for k, v in result.params.items()) or "defaults"
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result.strategy)} on {html.escape(coin)}</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:940px;margin:0 auto;padding:16px}}
svg{{width:100%;height:auto;border:1px solid var(--line);margin:8px 0}}
.t{{font-size:13px;fill:var(--fg)}} .a{{font-size:10px;fill:var(--mut)}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
.pos{{color:#1a9850}} .neg{{color:#d73027}} .mut{{color:var(--mut)}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>{html.escape(result.strategy)} on {html.escape(coin)} ({html.escape(interval)})</h1>
<p class="mut">Parameters: {html.escape(params)}. Leverage {result.config.leverage}x, taker fee {result.config.taker_fee * 100:.3f}%,
slippage {result.config.slippage_bps} bps. {eq.index[0]:%Y-%m-%d} to {eq.index[-1]:%Y-%m-%d}.</p>
{_chart([("strategy", eq, "#2b7bba"), ("buy and hold", bh, "#999999")], "Equity (USD)")}
{_chart([("drawdown %", dd, "#d73027")], "Drawdown (%)", h=160)}
<h2>Metrics</h2><div class="wrap"><table>{rows}</table></div>
<h2>Trades (last {len(trades)})</h2><div class="wrap"><table>
<tr><th>Entry</th><th>Side</th><th>Entry px</th><th>Exit</th><th>Exit px</th><th>Return</th><th>Exit reason</th></tr>
{trade_rows}</table></div>
<p class="mut">A backtest is a simulation of the past. It ignores order-book depth and latency, and parameters
chosen on this data are fitted to it. Use the walk-forward command for an out-of-sample estimate.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
