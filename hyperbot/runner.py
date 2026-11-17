"""Long-running loops: scheduling, and the rule-based (no LLM) strategy runner."""

from __future__ import annotations

import re
import signal
import time
from typing import Callable

from .config import Settings
from .data import interval_ms, load_history
from .risk import Executor
from .strategies import Strategy


def parse_duration(text: str) -> int:
    """'30s' / '15m' / '4h' / '1d' -> seconds."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", text.lower())
    if not m:
        raise ValueError(f"Bad duration {text!r}. Examples: 30s, 15m, 4h, 1d")
    return int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _raise_interrupt(signum, frame):  # docker stop sends SIGTERM
    raise KeyboardInterrupt


def run_forever(
    step: Callable[[], None],
    every_seconds: int,
    log: Callable[[str], None],
    align: bool = False,
    offset_seconds: int = 5,
) -> None:
    """Call `step` repeatedly until interrupted. Errors are logged, not fatal.

    With `align`, runs fire just after each multiple of the period (bar closes).
    """
    try:
        signal.signal(signal.SIGTERM, _raise_interrupt)
    except (ValueError, AttributeError):
        pass
    failures = 0
    try:
        while True:
            try:
                step()
                failures = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failures += 1
                log(f"cycle failed ({type(e).__name__}): {e}")
                if failures >= 10:
                    log("10 failures in a row, stopping")
                    raise
            now = time.time()
            if align:
                wait = every_seconds - (now % every_seconds) + offset_seconds
            else:
                wait = every_seconds
            log(f"next run in {int(wait)}s (Ctrl+C to stop)")
            time.sleep(wait)
    except KeyboardInterrupt:
        log("stopped; open positions are left as they are and keep their stop orders")


class StrategyRunner:
    """Trades one strategy on one coin: recompute the signal on each closed bar, then rebalance."""

    def __init__(
        self,
        settings: Settings,
        executor: Executor,
        strategy: Strategy,
        coin: str,
        interval: str,
        leverage: float = 1.0,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        warmup_bars: int = 800,
    ):
        self.s = settings
        self.executor = executor
        self.strategy = strategy
        self.coin = coin.upper()
        self.interval = interval
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct or settings.default_stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.days = warmup_bars * interval_ms(interval) / 86_400_000
        self.last_bar = None
        self.last_signal: float | None = None

    def step(self) -> dict:
        df = load_history(self.coin, self.interval, self.days, self.s.data_dir)
        bar = df.index[-1]
        if bar == self.last_bar:
            return {"status": "waiting", "detail": "no new closed bar yet"}
        sig = float(self.strategy.signal(df).iloc[-1])
        acct = self.executor.account()
        cur = acct.positions[self.coin].notional if self.coin in acct.positions else 0.0
        self.last_bar = bar
        # Trade when the signal changes, or when the position was closed underneath us (stop hit).
        stopped_out = self.last_signal is not None and sig == self.last_signal and sig != 0 and cur == 0
        if self.last_signal is not None and sig == self.last_signal and not stopped_out:
            return {"status": "hold", "signal": sig, "position_usd": round(cur, 2)}
        if stopped_out:
            return {"status": "stopped_out", "signal": sig, "detail": "flat until the signal changes"}
        self.last_signal = sig
        target = sig * self.leverage * acct.equity
        result = self.executor.set_target(
            self.coin,
            target,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            source=f"strategy:{self.strategy.name}",
            reason=f"signal {sig:+.2f} on {self.interval} bar {bar}",
        )
        return {"signal": sig, **result}
