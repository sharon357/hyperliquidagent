"""Brokers: a paper broker (simulated fills at real prices) and the live Hyperliquid broker."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .data import HLData


@dataclass
class Position:
    coin: str
    size: float  # signed units: positive long, negative short
    entry_px: float
    mark_px: float
    unrealized_pnl: float = 0.0
    liquidation_px: float | None = None

    @property
    def notional(self) -> float:
        """Signed USD value."""
        return self.size * self.mark_px


@dataclass
class AccountState:
    equity: float
    withdrawable: float
    margin_used: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.notional) for p in self.positions.values())

    def to_dict(self) -> dict:
        return {
            "equity_usd": round(self.equity, 2),
            "withdrawable_usd": round(self.withdrawable, 2),
            "margin_used_usd": round(self.margin_used, 2),
            "gross_exposure_usd": round(self.gross_exposure, 2),
            "leverage": round(self.gross_exposure / self.equity, 3) if self.equity > 0 else 0.0,
            "positions": [
                {
                    "coin": p.coin,
                    "side": "long" if p.size > 0 else "short",
                    "size": p.size,
                    "notional_usd": round(p.notional, 2),
                    "entry_px": p.entry_px,
                    "mark_px": p.mark_px,
                    "unrealized_pnl_usd": round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round((p.mark_px / p.entry_px - 1) * 100 * (1 if p.size > 0 else -1), 2)
                    if p.entry_px
                    else 0.0,
                    "liquidation_px": p.liquidation_px,
                }
                for p in self.positions.values()
            ],
        }


@dataclass
class Fill:
    ok: bool
    coin: str
    size: float = 0.0  # signed units actually traded
    price: float = 0.0
    fee: float = 0.0
    error: str = ""


class Broker:
    name = "broker"

    def account(self) -> AccountState:
        raise NotImplementedError

    def mids(self) -> dict[str, float]:
        raise NotImplementedError

    def market(self, coin: str, delta_units: float, reduce_only: bool = False) -> Fill:
        """Trade `delta_units` (signed) at market."""
        raise NotImplementedError

    def close(self, coin: str) -> Fill:
        raise NotImplementedError

    def set_protection(self, coin: str, stop_px: float | None, tp_px: float | None) -> list[str]:
        """Replace the stop-loss / take-profit orders for the coin's open position."""
        raise NotImplementedError

    def get_protection(self, coin: str) -> dict:
        return {}

    def cancel_all(self, coin: str) -> int:
        raise NotImplementedError

    def set_leverage(self, coin: str, leverage: int) -> None:
        pass


def walk_book(levels: list[tuple[float, float]], size: float) -> float | None:
    """Average price paid to take `size` units from one side of an order book.

    If the order is bigger than the visible book, the rest is assumed to fill
    progressively worse than the last visible level.
    """
    if not levels or size <= 0:
        return None
    remaining, cost = size, 0.0
    for px, sz in levels:
        take = min(remaining, sz)
        cost += take * px
        remaining -= take
        if remaining <= 0:
            return cost / size
    first, last = levels[0][0], levels[-1][0]
    visible = size - remaining
    # Extrapolate the book's slope: twice the size costs roughly twice the distance.
    extra = (last - first) * (remaining / max(visible, 1e-12))
    cost += remaining * (last + extra)
    return cost / size


def sqrt_impact_bps(notional_usd: float, daily_volume_usd: float, daily_vol_pct: float) -> float:
    """Square-root market-impact law: impact = daily volatility * sqrt(order / daily volume).

    This is the standard empirical estimate of how far an order moves the price
    beyond what the visible book shows (other traders pulling quotes or trading
    ahead of the rest of the order).
    """
    if notional_usd <= 0 or not daily_volume_usd or daily_volume_usd <= 0:
        return 0.0
    return daily_vol_pct * 100 * math.sqrt(notional_usd / daily_volume_usd)


class PaperBroker(Broker):
    """Simulated account trading against the real, live Hyperliquid market.

    A fill is priced in three layers:
    1. the real order book is walked for the order's size (spread + depth);
    2. a square-root impact term is added for the market's reaction to the order;
    3. our own recent trades in the same direction leave a "pressure" that decays
       over time and makes follow-up orders in that direction more expensive.
    Funding is charged on open positions at the live rate. Stops and take-profits
    are checked whenever the account is read, so they trigger at the next poll.

    What it cannot do: change the real price path. Other traders never see these
    orders, so their longer-term reactions (copying, fading, hunting stops) are
    not simulated.
    """

    name = "paper"

    def __init__(self, settings: Settings, data: HLData | None = None):
        self.s = settings
        self.data = data or HLData()
        self.path = Path(settings.state_dir) / "paper_account.json"
        self.state = self._load()
        self._ctx_cache: tuple[float, dict] = (0.0, {})
        self._vol_cache: dict[str, float] = {}

    def _fresh(self) -> dict:
        return {
            "cash": self.s.paper_equity,
            "start_equity": self.s.paper_equity,
            "positions": {},
            "protection": {},
            "fees_paid": 0.0,
            "costs": {"fees": 0.0, "spread_and_depth": 0.0, "impact": 0.0, "funding": 0.0},
            "pressure": {},
            "trades": 0,
            "volume_usd": 0.0,
        }

    def _load(self) -> dict:
        state = self._fresh()
        if self.path.is_file():
            state.update(json.loads(self.path.read_text(encoding="utf-8")))
        return state

    def _contexts(self) -> dict:
        """Live funding and volume per coin, cached for a minute."""
        stamp, ctxs = self._ctx_cache
        if time.time() - stamp > 60:
            ctxs = self.data.asset_contexts()
            self._ctx_cache = (time.time(), ctxs)
        return ctxs

    def _execution_price(self, coin: str, delta: float, mid: float) -> tuple[float, float, float]:
        """(fill price, spread+depth cost, impact cost) for trading `delta` units."""
        side = 1.0 if delta > 0 else -1.0
        size = abs(delta)
        book_px = None
        try:
            bids, asks = self.data.book_levels(coin)
            book_px = walk_book(asks if side > 0 else bids, size)
            if bids and asks:
                mid = (bids[0][0] + asks[0][0]) / 2  # measure costs against this same snapshot
        except Exception:
            pass  # no book available: fall back to the flat slippage setting
        if book_px is None:
            book_px = mid * (1 + side * self.s.slippage_bps / 1e4)

        impact_bps = 0.0
        try:
            if coin not in self._vol_cache:
                self._vol_cache[coin] = self.data.daily_vol_pct(coin)
            volume = self._contexts()[coin]["volume_24h_usd"]
            impact_bps = sqrt_impact_bps(size * mid, volume, self._vol_cache[coin]) * self.s.sim_impact
        except Exception:
            pass

        # Pressure left by our own earlier trades, decaying with a half-life.
        now = time.time()
        prev = self.state["pressure"].get(coin, {"bps": 0.0, "time": now})
        half_life = max(self.s.sim_impact_halflife_min * 60, 1.0)
        decayed = prev["bps"] * 0.5 ** ((now - prev["time"]) / half_life)
        residual_bps = max(0.0, decayed * side)  # only same-direction orders pay it; no rebate the other way
        self.state["pressure"][coin] = {"bps": decayed + side * impact_bps, "time": now}

        px = book_px * (1 + side * (impact_bps + residual_bps) / 1e4)
        return px, size * abs(book_px - mid), size * abs(px - book_px)

    def _accrue_funding(self, mids: dict[str, float]) -> None:
        now = time.time()
        last = self.state.get("funding_time", now)
        hours = (now - last) / 3600
        if not self.state["positions"]:
            self.state["funding_time"] = now
            return
        if "funding_time" in self.state and hours < 5 / 60:
            return
        try:
            ctxs = self._contexts()
        except Exception:
            return  # try again on the next poll; the elapsed time keeps accumulating
        for coin, p in self.state["positions"].items():
            rate = ctxs.get(coin, {}).get("funding_hourly", 0.0)
            if rate != rate:  # NaN
                continue
            pay = p["size"] * mids.get(coin, p["entry_px"]) * rate * min(hours, 24.0)
            self.state["cash"] -= pay  # longs pay when funding is positive
            self.state["costs"]["funding"] += pay
        self.state["funding_time"] = now
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.state = self._fresh()
        self._save()

    def mids(self) -> dict[str, float]:
        return self.data.mids()

    def _fill(self, coin: str, delta: float, mid: float) -> Fill:
        if delta == 0:
            return Fill(False, coin, error="zero size")
        px, depth_cost, impact_cost = self._execution_price(coin, delta, mid)
        fee = abs(delta) * px * self.s.taker_fee
        costs = self.state["costs"]
        costs["fees"] += fee
        costs["spread_and_depth"] += depth_cost
        costs["impact"] += impact_cost
        self.state["trades"] += 1
        self.state["volume_usd"] += abs(delta) * px
        pos = self.state["positions"].get(coin, {"size": 0.0, "entry_px": 0.0})
        size, entry = pos["size"], pos["entry_px"]
        new_size = size + delta
        realized = 0.0
        if size != 0 and (size > 0) != (delta > 0):  # reducing or flipping
            closed = min(abs(delta), abs(size))
            realized = closed * (px - entry) * (1 if size > 0 else -1)
            if abs(delta) > abs(size):
                entry = px  # flipped: the remainder opens at the fill price
        else:  # opening or adding
            entry = (entry * abs(size) + px * abs(delta)) / abs(new_size)
        self.state["cash"] += realized - fee
        self.state["fees_paid"] += fee
        if abs(new_size) * px < 0.01:
            self.state["positions"].pop(coin, None)
            self.state["protection"].pop(coin, None)
        else:
            self.state["positions"][coin] = {"size": new_size, "entry_px": entry}
        self._save()
        return Fill(True, coin, delta, px, fee)

    def _check_protection(self, mids: dict[str, float]) -> None:
        for coin, prot in list(self.state["protection"].items()):
            pos = self.state["positions"].get(coin)
            mid = mids.get(coin)
            if not pos or mid is None:
                self.state["protection"].pop(coin, None)
                continue
            long = pos["size"] > 0
            stop, tp = prot.get("stop_px"), prot.get("tp_px")
            hit_stop = stop is not None and (mid <= stop if long else mid >= stop)
            hit_tp = tp is not None and (mid >= tp if long else mid <= tp)
            if hit_stop or hit_tp:
                self._fill(coin, -pos["size"], mid)

    def account(self) -> AccountState:
        mids = self.mids()
        self._check_protection(mids)
        self._accrue_funding(mids)
        positions = {}
        unrealized = 0.0
        for coin, p in self.state["positions"].items():
            mark = mids.get(coin, p["entry_px"])
            upnl = p["size"] * (mark - p["entry_px"])
            unrealized += upnl
            positions[coin] = Position(coin, p["size"], p["entry_px"], mark, upnl)
        equity = self.state["cash"] + unrealized
        gross = sum(abs(p.notional) for p in positions.values())
        margin = gross / max(self.s.max_leverage, 1.0)
        return AccountState(equity, max(equity - margin, 0.0), margin, positions)

    def market(self, coin: str, delta_units: float, reduce_only: bool = False) -> Fill:
        mid = self.mids().get(coin)
        if mid is None:
            return Fill(False, coin, error=f"no price for {coin}")
        if reduce_only:
            size = self.state["positions"].get(coin, {}).get("size", 0.0)
            if size == 0 or (size > 0) == (delta_units > 0):
                return Fill(False, coin, error="reduce-only order would not reduce the position")
            if abs(delta_units) > abs(size):
                delta_units = -size
        return self._fill(coin, delta_units, mid)

    def close(self, coin: str) -> Fill:
        size = self.state["positions"].get(coin, {}).get("size", 0.0)
        if size == 0:
            return Fill(False, coin, error="no position")
        return self.market(coin, -size, reduce_only=True)

    def set_protection(self, coin: str, stop_px: float | None, tp_px: float | None) -> list[str]:
        if coin not in self.state["positions"]:
            return ["no position to protect"]
        self.state["protection"][coin] = {"stop_px": stop_px, "tp_px": tp_px}
        self._save()
        return []

    def get_protection(self, coin: str) -> dict:
        return dict(self.state["protection"].get(coin, {}))

    def cancel_all(self, coin: str) -> int:
        removed = 1 if self.state["protection"].pop(coin, None) else 0
        self._save()
        return removed


class HyperliquidBroker(Broker):
    """Live trading through the official hyperliquid-python-sdk (testnet or mainnet)."""

    def __init__(self, settings: Settings):
        if not settings.private_key:
            raise RuntimeError("HL_PRIVATE_KEY (or HL_PRIVATE_KEY_FILE) is required for testnet/mainnet")
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        self.s = settings
        self.name = settings.mode
        wallet = eth_account.Account.from_key(settings.private_key)
        # With an API (agent) wallet the key signs for the main account's address.
        self.address = settings.account_address or wallet.address
        self.info = Info(settings.api_url, skip_ws=True)
        self.exchange = Exchange(wallet, settings.api_url, account_address=self.address)
        universe = self.info.meta()["universe"]
        self._sz_decimals = {a["name"]: int(a["szDecimals"]) for a in universe}
        self._max_leverage = {a["name"]: int(a.get("maxLeverage", 1)) for a in universe}
        self._leverage_set: set[tuple[str, int]] = set()
        # Unified accounts (and portfolio margin) keep their USDC on the spot side and
        # use it directly as perp collateral, so the perp-only summary reads zero.
        try:
            self.abstraction = str(self.info.post("/info", {"type": "userAbstraction", "user": self.address}))
        except Exception:
            self.abstraction = "unknown"
        self.unified = self.abstraction in ("unifiedAccount", "portfolioMargin")

    def mids(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.info.all_mids().items()}

    def net_deposits(self) -> float | None:
        """USDC paid into the account minus USDC taken out, from the exchange's ledger (cached for a minute)."""
        cached = getattr(self, "_deposit_cache", None)
        if cached and time.time() - cached[0] < 60:
            return cached[1]
        try:
            ledger = self.info.post("/info", {"type": "userNonFundingLedgerUpdates", "user": self.address, "startTime": 0})
        except Exception:
            return cached[1] if cached else None
        me = self.address.lower()
        total = 0.0
        for entry in ledger:
            d = entry.get("delta", {})
            kind = d.get("type")
            if kind == "deposit":
                total += float(d.get("usdc", 0))
            elif kind == "withdraw":
                total -= float(d.get("usdc", 0)) + float(d.get("fee", 0))
            elif kind in ("send", "internalTransfer", "spotTransfer"):
                if d.get("token", "USDC") != "USDC":
                    continue
                amount = float(d.get("amount") or d.get("usdc") or 0)
                total += amount if str(d.get("destination", "")).lower() == me else -amount
        self._deposit_cache = (time.time(), total)
        return total

    def _spot_usdc(self) -> tuple[float, float]:
        """(total, on hold) USDC on the spot side."""
        for b in self.info.spot_user_state(self.address).get("balances", []):
            if b.get("coin") == "USDC":
                return float(b.get("total", 0)), float(b.get("hold", 0))
        return 0.0, 0.0

    def account(self) -> AccountState:
        st = self.info.user_state(self.address)
        ms = st["marginSummary"]
        positions = {}
        for ap in st.get("assetPositions", []):
            p = ap["position"]
            size = float(p["szi"])
            if size == 0:
                continue
            mark = abs(float(p["positionValue"]) / size)
            liq = p.get("liquidationPx")
            positions[p["coin"]] = Position(
                p["coin"],
                size,
                float(p.get("entryPx") or 0),
                mark,
                float(p.get("unrealizedPnl") or 0),
                float(liq) if liq else None,
            )
        equity = float(ms["accountValue"])
        margin = float(ms["totalMarginUsed"])
        free = float(st.get("withdrawable", 0))
        if self.unified:
            # Verified on a live account: the spot USDC total is the whole account value. It already
            # includes margin lent to open positions (reported as "hold") and their unrealised PnL,
            # so the perp summary must not be added on top.
            usdc, hold = self._spot_usdc()
            equity = usdc
            free = max(usdc - hold, 0.0)
        return AccountState(equity, free, margin, positions)

    def _round_size(self, coin: str, units: float) -> float:
        return round(abs(units), self._sz_decimals.get(coin, 4))

    def _round_price(self, coin: str, px: float) -> float:
        # Hyperliquid perps: at most 5 significant figures and (6 - szDecimals) decimals.
        return round(float(f"{px:.5g}"), 6 - self._sz_decimals.get(coin, 0))

    @staticmethod
    def _parse(coin: str, res, sign: float) -> Fill:
        if res is None:
            return Fill(False, coin, error="no position / nothing to do")
        if res.get("status") != "ok":
            return Fill(False, coin, error=str(res.get("response", res)))
        statuses = res["response"]["data"]["statuses"]
        size = notional = 0.0
        errors = []
        for st in statuses:
            if "filled" in st:
                sz = float(st["filled"]["totalSz"])
                size += sz
                notional += sz * float(st["filled"]["avgPx"])
            elif "error" in st:
                errors.append(st["error"])
        if size == 0:
            return Fill(False, coin, error="; ".join(errors) or "order did not fill")
        return Fill(True, coin, sign * size, notional / size, error="; ".join(errors))

    def set_leverage(self, coin: str, leverage: int) -> None:
        lev = max(1, min(int(leverage), self._max_leverage.get(coin, 1)))
        if (coin, lev) in self._leverage_set:
            return  # already set to this value; re-sent only when the limit changes
        self.exchange.update_leverage(lev, coin, is_cross=True)
        self._leverage_set = {x for x in self._leverage_set if x[0] != coin} | {(coin, lev)}

    def market(self, coin: str, delta_units: float, reduce_only: bool = False) -> Fill:
        sz = self._round_size(coin, delta_units)
        if sz <= 0:
            return Fill(False, coin, error="size rounds to zero")
        sign = 1.0 if delta_units > 0 else -1.0
        if reduce_only:
            res = self.exchange.market_close(coin, sz=sz, slippage=self.s.live_slippage)
        else:
            res = self.exchange.market_open(coin, delta_units > 0, sz, None, self.s.live_slippage)
        fill = self._parse(coin, res, sign)
        fill.fee = abs(fill.size) * fill.price * self.s.taker_fee  # estimate
        return fill

    def close(self, coin: str) -> Fill:
        pos = self.account().positions.get(coin)
        if not pos:
            return Fill(False, coin, error="no position")
        res = self.exchange.market_close(coin, slippage=self.s.live_slippage)
        return self._parse(coin, res, -1.0 if pos.size > 0 else 1.0)

    def set_protection(self, coin: str, stop_px: float | None, tp_px: float | None) -> list[str]:
        pos = self.account().positions.get(coin)
        if not pos:
            return ["no position to protect"]
        self.cancel_all(coin)
        errors = []
        closing_is_buy = pos.size < 0
        sz = self._round_size(coin, pos.size)
        for px, kind in ((stop_px, "sl"), (tp_px, "tp")):
            if px is None:
                continue
            px = self._round_price(coin, px)
            res = self.exchange.order(
                coin,
                closing_is_buy,
                sz,
                px,
                {"trigger": {"triggerPx": px, "isMarket": True, "tpsl": kind}},
                reduce_only=True,
            )
            if res.get("status") != "ok":
                errors.append(f"{kind}: {res.get('response', res)}")
            else:
                for st in res["response"]["data"]["statuses"]:
                    if isinstance(st, dict) and "error" in st:
                        errors.append(f"{kind}: {st['error']}")
        return errors

    def get_protection(self, coin: str) -> dict:
        out: dict = {}
        for o in self.info.frontend_open_orders(self.address):
            if o.get("coin") != coin or not o.get("isTrigger"):
                continue
            key = "stop_px" if "Stop" in str(o.get("orderType", "")) else "tp_px"
            out[key] = float(o.get("triggerPx") or 0)
        return out

    def cancel_all(self, coin: str) -> int:
        n = 0
        for o in self.info.frontend_open_orders(self.address):
            if o.get("coin") == coin:
                self.exchange.cancel(coin, o["oid"])
                n += 1
        return n


def make_broker(settings: Settings) -> Broker:
    if settings.mode == "paper":
        return PaperBroker(settings)
    return HyperliquidBroker(settings)
