"""Brokers: a paper broker (simulated fills at real prices) and the live Hyperliquid broker."""

from __future__ import annotations

import json
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


class PaperBroker(Broker):
    """Simulated account. Fills at the live Hyperliquid mid price plus slippage and taker fee.

    Stops and take-profits are checked whenever the account is read, so they trigger
    at the next poll rather than at the exact price. Funding is not simulated.
    """

    name = "paper"

    def __init__(self, settings: Settings, data: HLData | None = None):
        self.s = settings
        self.data = data or HLData()
        self.path = Path(settings.state_dir) / "paper_account.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"cash": self.s.paper_equity, "positions": {}, "protection": {}, "fees_paid": 0.0}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.state = {"cash": self.s.paper_equity, "positions": {}, "protection": {}, "fees_paid": 0.0}
        self._save()

    def mids(self) -> dict[str, float]:
        return self.data.mids()

    def _fill(self, coin: str, delta: float, mid: float) -> Fill:
        if delta == 0:
            return Fill(False, coin, error="zero size")
        slip = self.s.slippage_bps / 1e4
        px = mid * (1 + slip if delta > 0 else 1 - slip)
        fee = abs(delta) * px * self.s.taker_fee
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
        self._leverage_set: set[str] = set()

    def mids(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.info.all_mids().items()}

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
        return AccountState(
            float(ms["accountValue"]), float(st.get("withdrawable", 0)), float(ms["totalMarginUsed"]), positions
        )

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
        if coin in self._leverage_set:
            return
        lev = max(1, min(int(leverage), self._max_leverage.get(coin, 1)))
        self.exchange.update_leverage(lev, coin, is_cross=True)
        self._leverage_set.add(coin)

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
