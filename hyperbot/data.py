"""Market data: Hyperliquid public info API (no key needed), Binance klines, CSV cache."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .config import MAINNET_URL

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}
HOUR_MS = 3_600_000
OHLCV = ["open", "high", "low", "close", "volume"]
USER_AGENT = "hyperbot/0.1"


def now_ms() -> int:
    return int(time.time() * 1000)


def interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval!r}. Use one of {list(INTERVAL_MS)}")
    return INTERVAL_MS[interval]


def _request(method: str, url: str, retries: int = 4, **kw):
    """HTTP with retry on rate limits, server errors and network errors."""
    kw.setdefault("timeout", 20)
    kw.setdefault("headers", {"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code} from {url}")
            else:
                r.raise_for_status()
                return r
        except requests.RequestException as e:  # network error or 4xx
            if isinstance(e, requests.HTTPError):
                raise
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} tries: {last}")


class HLData:
    """Read-only Hyperliquid market data."""

    def __init__(self, base_url: str = MAINNET_URL):
        self.base_url = base_url.rstrip("/")

    def _post(self, payload: dict):
        return _request("POST", f"{self.base_url}/info", json=payload).json()

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        step = interval_ms(interval)
        rows: dict[int, dict] = {}
        cursor = start_ms
        while cursor < end_ms:
            batch = self._post(
                {
                    "type": "candleSnapshot",
                    "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": end_ms},
                }
            )
            if not batch:
                break
            for c in batch:
                rows[c["t"]] = c
            last_t = max(c["t"] for c in batch)
            nxt = last_t + step
            if nxt <= cursor or len(batch) < 2:
                break
            cursor = nxt
        if not rows:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC", name="time"))
        df = pd.DataFrame(
            {
                "time": pd.to_datetime([r["t"] for r in rows.values()], unit="ms", utc=True),
                "open": [float(r["o"]) for r in rows.values()],
                "high": [float(r["h"]) for r in rows.values()],
                "low": [float(r["l"]) for r in rows.values()],
                "close": [float(r["c"]) for r in rows.values()],
                "volume": [float(r["v"]) for r in rows.values()],
            }
        )
        return df.set_index("time").sort_index()

    def funding(self, coin: str, start_ms: int, end_ms: int) -> pd.Series:
        """Hourly funding rates (fraction per hour; positive = longs pay shorts)."""
        out: dict[int, float] = {}
        cursor = start_ms
        while cursor < end_ms:
            batch = self._post(
                {"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": end_ms}
            )
            if not batch:
                break
            for f in batch:
                out[f["time"]] = float(f["fundingRate"])
            last_t = max(f["time"] for f in batch)
            if last_t + 1 <= cursor or len(batch) < 2:
                break
            cursor = last_t + 1
        s = pd.Series(out, dtype=float).sort_index()
        s.index = pd.to_datetime(s.index, unit="ms", utc=True)
        s.index.name = "time"
        s.name = "funding"
        return s

    def mids(self) -> dict[str, float]:
        return {k: float(v) for k, v in self._post({"type": "allMids"}).items()}

    def meta(self) -> dict[str, dict]:
        """Per-coin static info: szDecimals, maxLeverage."""
        return {a["name"]: a for a in self._post({"type": "meta"})["universe"]}

    def asset_contexts(self) -> dict[str, dict]:
        """Per-coin live context: mark price, funding, open interest, 24h volume."""
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        out = {}
        for asset, ctx in zip(meta["universe"], ctxs):

            def num(key: str, ctx=ctx) -> float:
                v = ctx.get(key)
                return float(v) if v not in (None, "") else float("nan")

            mark, prev = num("markPx"), num("prevDayPx")
            out[asset["name"]] = {
                "mark_px": mark,
                "oracle_px": num("oraclePx"),
                "change_24h_pct": (mark / prev - 1) * 100 if prev and prev == prev else float("nan"),
                "funding_hourly": num("funding"),
                "funding_annual_pct": num("funding") * 24 * 365 * 100,
                "open_interest_usd": num("openInterest") * mark,
                "volume_24h_usd": num("dayNtlVlm"),
                "premium_pct": num("premium") * 100,
                "max_leverage": asset.get("maxLeverage"),
            }
        return out

    def orderbook(self, coin: str, depth: int = 10) -> dict:
        book = self._post({"type": "l2Book", "coin": coin})
        bids, asks = book["levels"][0][:depth], book["levels"][1][:depth]
        bid_sz = sum(float(b["sz"]) for b in bids)
        ask_sz = sum(float(a["sz"]) for a in asks)
        best_bid, best_ask = float(bids[0]["px"]), float(asks[0]["px"])
        mid = (best_bid + best_ask) / 2
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": (best_ask - best_bid) / mid * 1e4,
            "bid_size": bid_sz,
            "ask_size": ask_sz,
            "imbalance": (bid_sz - ask_sz) / (bid_sz + ask_sz) if bid_sz + ask_sz else 0.0,
        }


def binance_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """USDT-margined futures klines from Binance. Longer history than Hyperliquid keeps."""
    step = interval_ms(interval)
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _request(
            "GET",
            "https://fapi.binance.com/fapi/v1/klines",
            params={
                "symbol": f"{coin}USDT",
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
        ).json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        if len(batch) < 1500:
            break
    df = pd.DataFrame(
        {
            "time": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        }
    )
    return df.drop_duplicates("time").set_index("time").sort_index()


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df.set_index("time").sort_index()


def _cached_fetch(path: Path, fetch, start_ms: int, end_ms: int, step_ms: int, refresh: bool) -> pd.DataFrame:
    """Fetch [start, end], reusing and extending a CSV cache so history accumulates."""
    cached = None if refresh else _read_cache(path)
    if cached is None or cached.empty:
        df = fetch(start_ms, end_ms)
    else:
        parts = [cached]
        first = int(cached.index[0].timestamp() * 1000)
        last = int(cached.index[-1].timestamp() * 1000)
        if start_ms < first - step_ms:
            parts.append(fetch(start_ms, first))
        parts.append(fetch(last, end_ms))  # refetches the last bar, which may have been open
        df = pd.concat([p for p in parts if not p.empty])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    if not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return df


def attach_funding(df: pd.DataFrame, funding: pd.Series, interval: str) -> pd.DataFrame:
    """Add a `funding` column: the funding rate paid during each bar (summed)."""
    out = df.copy()
    if funding is None or funding.empty or df.empty:
        out["funding"] = 0.0
        return out
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    # A payment stamped 13:00 settles the 12:00-13:00 hour, so shift back 1 ms before flooring.
    bar_start = (funding.index - pd.Timedelta(milliseconds=1)).floor(step)
    per_bar = funding.groupby(bar_start).sum()
    out["funding"] = per_bar.reindex(out.index).fillna(0.0)
    return out


def load_history(
    coin: str,
    interval: str = "1h",
    days: float = 180,
    data_dir: Path = Path("data"),
    source: str = "hyperliquid",
    refresh: bool = False,
    closed_only: bool = True,
    with_funding: bool = True,
) -> pd.DataFrame:
    """OHLCV + funding for a coin. Index is the UTC bar open time."""
    coin = coin.upper()
    step = interval_ms(interval)
    end = now_ms()
    start = end - int(days * 86_400_000)
    hl = HLData()
    if source == "hyperliquid":
        fetch = lambda a, b: hl.candles(coin, interval, a, b)  # noqa: E731
    elif source == "binance":
        fetch = lambda a, b: binance_candles(coin, interval, a, b)  # noqa: E731
    else:
        raise ValueError("source must be 'hyperliquid' or 'binance'")
    df = _cached_fetch(data_dir / f"{source}_{coin}_{interval}.csv", fetch, start, end, step, refresh)
    if df.empty:
        raise RuntimeError(f"No candles returned for {coin} {interval} from {source}")
    df = df[df.index >= pd.Timestamp(start, unit="ms", tz="UTC")]
    if closed_only:
        df = df[df.index + pd.Timedelta(milliseconds=step) <= pd.Timestamp(end, unit="ms", tz="UTC")]

    funding = None
    if with_funding:
        def fetch_funding(a: int, b: int) -> pd.DataFrame:
            return hl.funding(coin, a, b).to_frame()

        fdf = _cached_fetch(data_dir / f"funding_{coin}.csv", fetch_funding, start, end, HOUR_MS, refresh)
        funding = fdf["funding"] if not fdf.empty else None
    return attach_funding(df, funding, interval)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load your own OHLCV file. Needs time/open/high/low/close columns; volume and funding optional."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    tcol = next((c for c in ("time", "timestamp", "date", "datetime", "open_time") if c in df.columns), None)
    if tcol is None:
        raise ValueError("CSV needs a time column (time/timestamp/date/datetime/open_time)")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        df["time"] = pd.to_datetime(t, unit=unit, utc=True)
    else:
        df["time"] = pd.to_datetime(t, utc=True)
    df = df.set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"CSV is missing the {c!r} column")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "funding" not in df.columns:
        df["funding"] = 0.0
    return df[OHLCV + ["funding"]].astype(float)


def infer_interval_ms(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return HOUR_MS
    return int(df.index.to_series().diff().dropna().median().total_seconds() * 1000)
