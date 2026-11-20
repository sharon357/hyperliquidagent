"""Hard risk limits and order execution.

Everything that trades (the LLM agent, the strategy runner, the CLI) goes through
`Executor.set_target`. The limits here are plain code: the agent cannot talk its
way past them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .exchange import AccountState, Broker


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# The AI manager may move the risk limits, but only inside these outer bounds. They are
# deliberately not adjustable by the model: text it reads from the internet could try to
# talk it into switching protection off, and this is what still stands if that works.
LIMIT_BOUNDS = {
    "max_leverage": (1.0, 20.0),
    "max_position_pct": (50.0, 2000.0),
    "max_alt_position_pct": (25.0, 2000.0),
    "max_daily_loss_pct": (2.0, 50.0),
    "max_drawdown_pct": (5.0, 80.0),
    "max_trades_per_day": (5, 100),
    "max_stop_loss_pct": (2.0, 50.0),
}


def _overrides_path(settings: Settings) -> Path:
    return Path(settings.state_dir) / "risk_overrides.json"


def apply_limit_overrides(settings: Settings) -> dict:
    """Load limits the AI manager set earlier onto the live settings. Returns what was applied."""
    p = _overrides_path(settings)
    if not settings.ai_sets_limits or not p.is_file():
        return {}
    saved = json.loads(p.read_text(encoding="utf-8"))
    for key, value in saved.get("limits", {}).items():
        if key in LIMIT_BOUNDS or key == "require_stop":
            setattr(settings, key, value)
    return saved


def set_limit_overrides(settings: Settings, requested: dict, rationale: str = "") -> dict:
    """Change risk limits at the AI manager's request, clamped to LIMIT_BOUNDS."""
    if not settings.ai_sets_limits:
        return {"error": "the owner has fixed the risk limits (HL_AI_SETS_LIMITS is off)"}
    applied, notes = {}, []
    for key, (lo, hi) in LIMIT_BOUNDS.items():
        if requested.get(key) in (None, "", 0):
            continue
        try:
            value = float(requested[key])
        except (TypeError, ValueError):
            notes.append(f"{key}: not a number, ignored")
            continue
        clamped = min(max(value, lo), hi)
        if clamped != value:
            notes.append(f"{key}: {value:g} is outside the allowed range, set to {clamped:g}")
        applied[key] = int(clamped) if isinstance(lo, int) else clamped
    if isinstance(requested.get("require_stop"), bool):
        applied["require_stop"] = requested["require_stop"]
    for key, value in applied.items():
        setattr(settings, key, value)
    p = _overrides_path(settings)
    saved = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"limits": {}}
    saved["limits"].update(applied)
    saved.update(updated=utc_now().isoformat(timespec="seconds"), rationale=str(rationale)[:800])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return {"status": "saved", "applied": applied, "notes": notes, "outer_bounds": LIMIT_BOUNDS}


class AICostLedger:
    """Running total of what the LLM has cost, kept next to the account it trades."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.path = Path(settings.state_dir) / "llm_usage.json"

    def load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"calls": 0, "cycles": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def price(self, prompt: int, cached: int, completion: int) -> float:
        s = self.s
        fresh = max(prompt - cached, 0)
        return (fresh * s.llm_price_input + cached * s.llm_price_input_cached + completion * s.llm_price_output) / 1e6

    def add(self, prompt: int, cached: int, completion: int, calls: int) -> float:
        """Record one decision cycle's usage. Returns that cycle's cost in USD."""
        cost = self.price(prompt, cached, completion)
        d = self.load()
        d["calls"] += calls
        d["cycles"] += 1
        d["prompt_tokens"] += prompt
        d["cached_tokens"] += cached
        d["completion_tokens"] += completion
        d["cost_usd"] = round(d["cost_usd"] + cost, 6)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return cost

    def total(self) -> float:
        return float(self.load()["cost_usd"])


class Journal:
    """Append-only JSONL log of decisions and trades."""

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "journal.jsonl"

    def write(self, kind: str, **fields) -> dict:
        rec = {"time": utc_now().isoformat(timespec="seconds"), "kind": kind, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return rec

    def tail(self, n: int = 20, kind: str | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or rec.get("kind") == kind:
                out.append(rec)
        return out[-n:]


@dataclass
class RiskDecision:
    approved: bool
    target_usd: float
    notes: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings
        self.path = Path(settings.state_dir) / "risk_state.json"
        self.state = self._load()
        apply_limit_overrides(settings)

    def _load(self) -> dict:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"day": "", "day_start_equity": 0.0, "high_water_mark": 0.0, "trades_today": 0, "halted": False, "halt_reason": "", "baseline_equity": 0.0}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def update(self, equity: float, net_deposits: float | None = None) -> None:
        """Roll the daily counters and trip the kill switch on a drawdown breach.

        `net_deposits` (live accounts) is everything paid in minus everything taken out.
        When it changes, the day's starting balance and the high-water mark move by the
        same amount, so a deposit is not read as profit and a withdrawal is not read as a loss.
        """
        if net_deposits is not None:
            if "net_deposits" not in self.state:
                # First time this account is tracked with deposits: restart the reference points here.
                self.state.update(net_deposits=net_deposits, day_start_equity=equity)
                self.state["high_water_mark"] = max(self.state.get("high_water_mark", 0.0), equity)
            elif abs(net_deposits - self.state["net_deposits"]) > 1e-6:
                moved = net_deposits - self.state["net_deposits"]
                self.state["net_deposits"] = net_deposits
                self.state["day_start_equity"] = max(self.state["day_start_equity"] + moved, 0.0)
                self.state["high_water_mark"] = max(self.state["high_water_mark"] + moved, 0.0)
        today = utc_now().strftime("%Y-%m-%d")
        if not self.state.get("baseline_equity") and equity > 0:
            # Equity when the bot first saw the account: the starting point for profit and loss.
            self.state.update(baseline_equity=equity, baseline_time=utc_now().isoformat(timespec="seconds"))
        if self.state["day"] != today:
            self.state.update(day=today, day_start_equity=equity, trades_today=0)
        if equity > self.state["high_water_mark"]:
            self.state["high_water_mark"] = equity
        hwm = self.state["high_water_mark"]
        if hwm > 0 and not self.state["halted"]:
            dd = (1 - equity / hwm) * 100
            if dd >= self.s.max_drawdown_pct:
                self.halt(f"drawdown {dd:.1f}% from the high-water mark reached the {self.s.max_drawdown_pct}% limit")
        self._save()

    def halt(self, reason: str) -> None:
        self.state.update(halted=True, halt_reason=reason)
        self._save()

    def resume(self, equity: float | None = None) -> None:
        self.state.update(halted=False, halt_reason="")
        if equity is not None:
            self.state["high_water_mark"] = equity  # restart drawdown tracking from here
        self._save()

    def pnl(self, equity: float, net_deposits: float | None = None) -> dict:
        """Profit and loss, before and after the LLM bill.

        With `net_deposits` (live accounts) profit is equity minus everything paid in,
        so topping up the account is not mistaken for a gain.
        """
        base = net_deposits if net_deposits else (self.state.get("baseline_equity") or equity)
        gross = equity - base
        ai = AICostLedger(self.s).total()
        net = gross - ai
        return {
            "since": self.state.get("baseline_time", ""),
            "starting_equity_usd": round(base, 2),
            "trading_pnl_usd": round(gross, 2),
            "ai_cost_usd": round(ai, 4),
            "net_pnl_usd": round(net, 2),
            "net_return_pct": round(net / base * 100, 3) if base else 0.0,
            "note": "trading PnL already includes fees, slippage and funding; deposits or withdrawals after the start distort it",
        }

    def daily_pnl_pct(self, equity: float) -> float:
        start = self.state["day_start_equity"]
        return (equity / start - 1) * 100 if start > 0 else 0.0

    def record_trade(self) -> None:
        self.state["trades_today"] += 1
        self._save()

    def status(self, equity: float) -> dict:
        hwm = self.state["high_water_mark"]
        return {
            "halted": self.state["halted"],
            "halt_reason": self.state["halt_reason"],
            "daily_pnl_pct": round(self.daily_pnl_pct(equity), 2),
            "drawdown_pct": round((1 - equity / hwm) * 100, 2) if hwm > 0 else 0.0,
            "trades_today": self.state["trades_today"],
            "limits": self.limits(),
        }

    def limits(self) -> dict:
        s = self.s
        return {
            "allowed_coins": s.coins,
            "max_leverage": s.max_leverage,
            "max_position_pct_of_equity": s.max_position_pct,
            "max_position_pct_for_non_core_coins": min(s.max_position_pct, s.max_alt_position_pct),
            "core_coins": s.core_coins or s.coins,
            "max_daily_loss_pct": s.max_daily_loss_pct,
            "max_drawdown_pct": s.max_drawdown_pct,
            "max_trades_per_day": s.max_trades_per_day,
            "min_order_usd": s.min_order_usd,
            "max_stop_loss_pct": s.max_stop_loss_pct,
            "stop_loss_required": s.require_stop,
            "default_trailing": (
                f"trail and break-even at {s.default_trail_mult:g}x the stop distance unless a trade sets its own"
                if s.default_trail_mult > 0
                else "none unless a trade sets its own"
            ),
            "owner_profit_target": (
                f"{s.profit_target_pct:g}% of the account per day, banked: the engine closes every position each time the "
                f"account has gained {s.profit_target_pct:g}% since the last bank, then calls you to rebuild from flat"
                if s.profit_target_pct > 0
                else "none"
            ),
            "owner_trailing_cap": (
                f"every position trails at most {s.max_trail_pct:g}% behind its best price and moves to break-even "
                f"once {s.max_trail_pct:g}% in profit at the latest; a trade may set tighter values, never wider or off"
                if s.max_trail_pct > 0
                else "none"
            ),
            "limits_set_by": "the AI manager, inside fixed outer bounds" if s.ai_sets_limits else "the owner",
            "autopilot_exposure_multiplier": s.exposure_multiplier,
        }

    def check(self, coin: str, target_usd: float, account: AccountState) -> RiskDecision:
        """Approve, shrink or reject a requested position (signed USD notional)."""
        s = self.s
        notes: list[str] = []
        equity = account.equity
        cur = account.positions[coin].notional if coin in account.positions else 0.0
        reducing = abs(target_usd) <= abs(cur) and (target_usd == 0 or (target_usd > 0) == (cur > 0))
        if reducing:
            return RiskDecision(True, target_usd, ["risk-reducing: always allowed"])
        if coin not in s.coins:
            return RiskDecision(False, cur, [f"{coin} is not in the allowed coins {s.coins}"])
        if self.state["halted"]:
            return RiskDecision(False, cur, [f"trading is halted: {self.state['halt_reason']}"])
        if equity <= 0:
            return RiskDecision(False, cur, ["account equity is zero"])
        if self.daily_pnl_pct(equity) <= -s.max_daily_loss_pct:
            return RiskDecision(False, cur, [f"daily loss limit of {s.max_daily_loss_pct}% reached; only reductions allowed today"])
        if self.state["trades_today"] >= s.max_trades_per_day:
            return RiskDecision(False, cur, [f"daily trade limit of {s.max_trades_per_day} reached"])

        sign = 1 if target_usd > 0 else -1
        size = abs(target_usd)
        cap_pct = s.max_position_pct
        if s.core_coins and coin not in s.core_coins:
            cap_pct = min(cap_pct, s.max_alt_position_pct)  # smaller, thinner coins get a tighter cap
        per_coin_cap = equity * cap_pct / 100
        if size > per_coin_cap:
            notes.append(f"clamped to the per-coin cap of ${per_coin_cap:,.0f}")
            size = per_coin_cap
        other = account.gross_exposure - abs(cur)
        gross_cap = equity * s.max_leverage - other
        if size > gross_cap:
            notes.append(f"clamped to keep total leverage at or below {s.max_leverage}x")
            size = max(gross_cap, 0.0)
        if size < s.min_order_usd:
            return RiskDecision(False, cur, notes + ["nothing left to trade after applying limits"])
        return RiskDecision(True, sign * size, notes)


class Executor:
    """Moves a coin's position to a target notional, subject to the risk manager."""

    def __init__(self, settings: Settings, broker: Broker, risk: RiskManager, journal: Journal, dry_run: bool = False):
        self.s = settings
        self.broker = broker
        self.risk = risk
        self.journal = journal
        self.dry_run = dry_run

    def _deposits(self) -> float | None:
        return getattr(self.broker, "net_deposits", lambda: None)()

    def account(self) -> AccountState:
        acct = self.broker.account()
        self.risk.update(acct.equity, self._deposits())
        return acct

    def pnl_report(self, equity: float) -> dict:
        return self.risk.pnl(equity, self._deposits())

    def set_target(
        self,
        coin: str,
        target_usd: float,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        source: str = "manual",
        reason: str = "",
        refresh_protection: bool = True,
    ) -> dict:
        """Trade so that `coin` holds `target_usd` of signed notional.

        Stop-loss and take-profit are percentages from the current mark price.
        """
        coin = coin.upper()
        s = self.s
        acct = self.account()
        cur = acct.positions[coin].notional if coin in acct.positions else 0.0
        result: dict = {"coin": coin, "requested_target_usd": round(target_usd, 2), "current_usd": round(cur, 2)}

        if target_usd != 0:
            if not stop_loss_pct or stop_loss_pct <= 0:
                if s.require_stop:
                    stop_loss_pct = s.default_stop_loss_pct
                    result["note_stop"] = f"no stop given, using the default {stop_loss_pct}%"
                else:
                    stop_loss_pct = None
                    result["note_stop"] = "no stop-loss on this position"
            if stop_loss_pct and stop_loss_pct > s.max_stop_loss_pct:
                stop_loss_pct = s.max_stop_loss_pct
                result["note_stop"] = f"stop clamped to the maximum {stop_loss_pct}%"

        decision = self.risk.check(coin, target_usd, acct)
        result["risk_notes"] = decision.notes
        if not decision.approved:
            result.update(status="rejected")
            self.journal.write("rejected", source=source, reason=reason, **result)
            return result
        target = decision.target_usd
        delta = target - cur
        result["approved_target_usd"] = round(target, 2)

        mid = self.broker.mids().get(coin)
        if not mid:
            result.update(status="error", error=f"no price for {coin}")
            return result

        flip = cur != 0 and target != 0 and (cur > 0) != (target > 0)
        small = abs(delta) < max(s.min_order_usd, s.rebalance_band * abs(cur))
        trade_needed = target == 0 and cur != 0 or flip or not small
        if target == 0 and cur == 0:
            result.update(status="noop", detail="already flat")
            return result

        if not trade_needed and not refresh_protection and (
            not stop_loss_pct or self.broker.get_protection(coin).get("stop_px")
        ):
            # Leave the existing stop where it is: re-placing it every tick would
            # drag it away from a falling price and it would never trigger.
            result.update(status="hold")
            return result

        if self.dry_run:
            result.update(status="dry_run", would_trade_usd=round(delta, 2) if trade_needed else 0.0)
            self.journal.write("dry_run", source=source, reason=reason, **result)
            return result

        fills = []
        if trade_needed:
            self.broker.cancel_all(coin)
            if target == 0 or flip:
                f = self.broker.close(coin)
                fills.append(f)
                if not f.ok:
                    result.update(status="error", error=f"close failed: {f.error}")
                    self.journal.write("error", source=source, reason=reason, **result)
                    return result
                if flip:
                    self.broker.set_leverage(coin, int(s.max_leverage))
                    fills.append(self.broker.market(coin, target / mid))
            elif abs(target) < abs(cur):
                fills.append(self.broker.market(coin, delta / mid, reduce_only=True))
            else:
                self.broker.set_leverage(coin, int(s.max_leverage))
                fills.append(self.broker.market(coin, delta / mid))
            failed = [f for f in fills if not f.ok]
            if failed and len(failed) == len(fills):
                result.update(status="error", error=failed[0].error)
                self.journal.write("error", source=source, reason=reason, **result)
                return result
            self.risk.record_trade()
            result["fills"] = [
                {"size": f.size, "price": f.price, "usd": round(f.size * f.price, 2), "ok": f.ok, "error": f.error}
                for f in fills
            ]
        else:
            result["detail"] = "position already within the rebalance band; no trade"

        # (Re)place protection on whatever position is now open.
        if target != 0:
            pos = self.broker.account().positions.get(coin)
            if pos:
                long = pos.size > 0
                stop_px = None
                if stop_loss_pct:
                    stop_px = pos.mark_px * (1 - stop_loss_pct / 100 if long else 1 + stop_loss_pct / 100)
                tp_px = None
                if take_profit_pct and take_profit_pct > 0:
                    tp_px = pos.mark_px * (1 + take_profit_pct / 100 if long else 1 - take_profit_pct / 100)
                errors = self.broker.set_protection(coin, stop_px, tp_px)
                result["stop_px"] = round(stop_px, 6) if stop_px else None
                result["take_profit_px"] = round(tp_px, 6) if tp_px else None
                if errors:
                    result["protection_errors"] = errors
        result["status"] = "executed" if trade_needed else "protection_updated"
        self.journal.write("trade" if trade_needed else "protection", source=source, reason=reason, **result)
        return result

    def flatten_all(self, source: str = "manual", reason: str = "") -> list[dict]:
        out = []
        for coin in list(self.account().positions):
            out.append(self.set_target(coin, 0.0, source=source, reason=reason))
        return out
