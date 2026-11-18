"""Which perps the bot may trade.

In `auto` universe mode the bot is not limited to a fixed list: it scans every
Hyperliquid perp, keeps the ones liquid enough to trade, and researches the core
coins plus a shortlist of the busiest and fastest-moving others. The manager
agent can pull further coins in by researching them.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .data import HLData

MAX_UNIVERSE = 12


def scan(data: HLData, min_volume_usd: float) -> list[dict]:
    """Every listed perp with at least `min_volume_usd` traded in 24h, busiest first."""
    rows = []
    for coin, c in data.asset_contexts().items():
        vol = c.get("volume_24h_usd")
        if c.get("delisted") or vol != vol or vol < min_volume_usd:
            continue
        chg = c.get("change_24h_pct")
        rows.append(
            {
                "coin": coin,
                "price": c["mark_px"],
                "change_24h_pct": round(chg, 2) if chg == chg else 0.0,
                "volume_24h_usd": round(vol),
                "open_interest_usd": round(c["open_interest_usd"]) if c["open_interest_usd"] == c["open_interest_usd"] else 0,
                "funding_annual_pct": round(c["funding_annual_pct"], 1) if c["funding_annual_pct"] == c["funding_annual_pct"] else 0.0,
                "max_leverage": c.get("max_leverage"),
            }
        )
    rows.sort(key=lambda r: r["volume_24h_usd"], reverse=True)
    return rows


def shortlist(rows: list[dict], core: list[str], extra: int) -> list[str]:
    """Core coins plus `extra` others: half the biggest movers, half the most traded."""
    out = [c for c in core]
    others = [r for r in rows if r["coin"] not in out]
    movers = sorted(others, key=lambda r: abs(r["change_24h_pct"]), reverse=True)
    busiest = others  # already sorted by volume
    take_movers = (extra + 1) // 2
    for r in movers[:take_movers]:
        out.append(r["coin"])
    for r in busiest:
        if len(out) >= len(core) + extra:
            break
        if r["coin"] not in out:
            out.append(r["coin"])
    return out[:MAX_UNIVERSE]


def _path(settings: Settings) -> Path:
    return Path(settings.state_dir) / "universe.json"


def save(settings: Settings) -> None:
    p = _path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"coins": list(settings.coins)}), encoding="utf-8")


def restore(settings: Settings) -> None:
    """Bring back the coin list chosen before a restart (only in auto mode)."""
    p = _path(settings)
    if settings.universe_mode != "auto" or not p.is_file():
        return
    coins = json.loads(p.read_text(encoding="utf-8")).get("coins") or []
    merged = list(dict.fromkeys([*settings.core_coins, *coins]))[:MAX_UNIVERSE]
    settings.coins[:] = merged  # in place: every component shares this list
