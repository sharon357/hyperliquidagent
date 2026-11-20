"""Market sentiment from free public sources.

- Crypto Fear & Greed index (alternative.me)
- News headlines from RSS feeds, with a simple keyword score
- Positioning on Hyperliquid itself: funding, open interest, premium

Headlines are untrusted text from the internet. They are cleaned and truncated
here, and the agent is told to treat them as data, never as instructions.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from .data import HLData, _request

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

BULLISH = {
    "surge", "surges", "rally", "rallies", "soar", "soars", "jump", "jumps", "gain", "gains", "bullish",
    "record", "high", "approval", "approved", "approves", "adoption", "inflow", "inflows", "buy", "buys",
    "upgrade", "breakout", "rebound", "rebounds", "recover", "recovers", "accumulate", "partnership", "launch",
    "launches", "rise", "rises", "boost", "boosts", "optimism", "etf",
}
BEARISH = {
    "crash", "crashes", "plunge", "plunges", "drop", "drops", "fall", "falls", "slump", "slumps", "bearish",
    "hack", "hacked", "exploit", "exploited", "lawsuit", "sues", "sue", "ban", "bans", "outflow", "outflows",
    "sell", "selloff", "sell-off", "liquidation", "liquidations", "liquidated", "fraud", "scam", "collapse",
    "bankruptcy", "bankrupt", "fear", "warning", "warns", "decline", "declines", "tumble", "tumbles", "dump",
    "dumps", "investigation", "probe", "delist", "delists", "losses", "loss", "rug",
}
COIN_NAMES = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "HYPE": ("hyperliquid", "hype"),
    "XRP": ("xrp", "ripple"),
    "DOGE": ("dogecoin", "doge"),
    "BNB": ("bnb", "binance"),
}


def clean_text(text: str, limit: int = 180) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def lexicon_score(title: str) -> int:
    """Crude keyword score: positive = bullish words, negative = bearish words."""
    words = re.findall(r"[a-z][a-z\-]+", title.lower())
    return sum(w in BULLISH for w in words) - sum(w in BEARISH for w in words)


def fear_greed() -> dict:
    data = _request("GET", "https://api.alternative.me/fng/", params={"limit": 8}).json()["data"]
    values = [int(d["value"]) for d in data]
    return {
        "value": values[0],
        "label": data[0]["value_classification"],
        "week_ago": values[-1],
        "change_7d": values[0] - values[-1],
        "history_newest_first": values,
        "scale": "0 = extreme fear, 100 = extreme greed",
    }


def news(limit: int = 30, coins: list[str] | None = None) -> list[dict]:
    items = []
    for source, url in FEEDS.items():
        try:
            root = ET.fromstring(_request("GET", url, retries=2, timeout=12).content)
        except Exception:
            continue  # one dead feed should not break the snapshot
        for item in root.iter("item"):
            title = clean_text(item.findtext("title") or "")
            if not title:
                continue
            published = ""
            try:
                published = parsedate_to_datetime(item.findtext("pubDate") or "").isoformat(timespec="minutes")
            except (TypeError, ValueError):
                pass
            lower = title.lower()
            tagged = [c for c, names in COIN_NAMES.items() if any(re.search(rf"\b{n}\b", lower) for n in names)]
            items.append(
                {"source": source, "published": published, "title": title, "coins": tagged, "keyword_score": lexicon_score(title)}
            )
    items.sort(key=lambda x: x["published"], reverse=True)
    if coins:
        wanted = set(coins)
        # Coin-specific headlines first, then general market news.
        items.sort(key=lambda x: (not (wanted & set(x["coins"])),))
    return items[:limit]


def positioning(data: HLData, coins: list[str]) -> dict:
    ctxs = data.asset_contexts()
    out = {}
    for c in coins:
        ctx = ctxs.get(c)
        if not ctx:
            continue
        out[c] = {
            "funding_annual_pct": round(ctx["funding_annual_pct"], 2),
            "premium_pct": round(ctx["premium_pct"], 4),
            "open_interest_usd": round(ctx["open_interest_usd"]),
            "volume_24h_usd": round(ctx["volume_24h_usd"]),
            "change_24h_pct": round(ctx["change_24h_pct"], 2),
        }
    return out


def snapshot(coins: list[str], data: HLData | None = None, news_limit: int = 30) -> dict:
    """Everything in one dict, plus a rough composite score in [-1, 1]."""
    out: dict = {"errors": []}
    parts = []
    try:
        fg = fear_greed()
        out["fear_greed"] = fg
        parts.append((fg["value"] - 50) / 50)
    except Exception as e:
        out["errors"].append(f"fear_greed: {e}")
    try:
        headlines = news(news_limit, coins)
        out["headlines"] = headlines
        if headlines:
            scores = [max(-1, min(1, h["keyword_score"])) for h in headlines]
            out["news_keyword_score"] = round(sum(scores) / len(scores), 3)
            parts.append(out["news_keyword_score"])
    except Exception as e:
        out["errors"].append(f"news: {e}")
    try:
        pos = positioning(data or HLData(), coins)
        out["positioning"] = pos
        if pos:
            # Funding of +50 % APR or more counts as fully crowded long.
            avg = sum(max(-1, min(1, p["funding_annual_pct"] / 50)) for p in pos.values()) / len(pos)
            out["funding_crowding_score"] = round(avg, 3)
            parts.append(avg)
    except Exception as e:
        out["errors"].append(f"positioning: {e}")
    out["composite_score"] = round(sum(parts) / len(parts), 3) if parts else None
    out["note"] = (
        "composite_score is a rough average of the parts (-1 bearish/fearful, +1 bullish/greedy). "
        "Extremes are often contrarian: extreme greed and crowded funding tend to precede pullbacks."
    )
    return out


def search_news(query: str, limit: int = 12) -> list[dict]:
    """Search recent news across the web for a query (Google News). Untrusted text."""
    from urllib.parse import quote_plus

    url = f"https://news.google.com/rss/search?q={quote_plus(query[:120])}+when:7d&hl=en-US&gl=US&ceid=US:en"
    root = ET.fromstring(_request("GET", url, retries=2, timeout=15).content)
    items = []
    for item in root.iter("item"):
        title = clean_text(item.findtext("title") or "", 200)
        if not title:
            continue
        published = ""
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").isoformat(timespec="minutes")
        except (TypeError, ValueError):
            pass
        items.append(
            {
                "published": published,
                "source": clean_text(item.findtext("source") or "", 60),
                "title": title,
                "keyword_score": lexicon_score(title),
            }
        )
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]


def trending() -> list[dict]:
    """What retail is searching for right now (CoinGecko trending). A crowd-attention gauge."""
    data = _request("GET", "https://api.coingecko.com/api/v3/search/trending", retries=2, timeout=15).json()
    out = []
    for c in data.get("coins", [])[:15]:
        item = c.get("item", {})
        change = (item.get("data", {}).get("price_change_percentage_24h") or {}).get("usd")
        out.append(
            {
                "symbol": str(item.get("symbol", "")).upper()[:12],
                "name": clean_text(item.get("name") or "", 40),
                "market_cap_rank": item.get("market_cap_rank"),
                "change_24h_pct": round(change, 1) if isinstance(change, (int, float)) else None,
            }
        )
    return out
