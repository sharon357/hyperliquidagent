"""Command-line interface: `hyperbot --help`."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backtest import BacktestConfig, grid_search, run_backtest, walk_forward
from .config import Settings, load_settings
from .data import HLData, load_csv, load_history
from .strategies import REGISTRY, get_strategy

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Hyperliquid trading bot: backtests, strategies, risk engine and an LLM decision agent.",
)
console = Console()

# Shared option definitions
O_INTERVAL = typer.Option("1h", "--interval", "-i", help="Candle size: 15m, 1h, 4h, 1d ...")
O_DAYS = typer.Option(180.0, "--days", "-d", help="How much history to use")
O_SOURCE = typer.Option("hyperliquid", "--source", help="hyperliquid or binance (longer history)")
O_CSV = typer.Option(None, "--csv", help="Use your own OHLCV CSV instead of downloading")
O_LEVERAGE = typer.Option(1.0, "--leverage", "-l", help="Notional as a multiple of equity")
O_PARAM = typer.Option(None, "--param", "-p", help="Strategy parameter, e.g. -p fast=10 (repeatable)")
O_SL = typer.Option(None, "--stop-loss", help="Stop-loss in percent")
O_TP = typer.Option(None, "--take-profit", help="Take-profit in percent")
O_LIVE = typer.Option(False, "--live", help="Required to trade real money when HL_MODE=mainnet")
O_DRY = typer.Option(False, "--dry-run", help="Decide and log, but place no orders")


def _value(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _params(items: Optional[list[str]]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"--param needs key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _value(v.strip())
    return out


def _load(settings: Settings, coin: str, interval: str, days: float, source: str, csv: Optional[str]) -> pd.DataFrame:
    if csv:
        return load_csv(csv)
    with console.status(f"Loading {coin} {interval} candles..."):
        df = load_history(coin, interval, days, settings.data_dir, source)
    got = (df.index[-1] - df.index[0]).days
    if got < days * 0.8:
        console.print(
            f"[yellow]Only {got} days of {interval} history available from {source} "
            f"(asked for {days:g}). Hyperliquid keeps about 5000 candles per interval; "
            "try --source binance or a larger --interval.[/yellow]"
        )
    return df


def _cfg(
    settings: Settings,
    leverage: float,
    sl: Optional[float],
    tp: Optional[float],
    trail: Optional[float] = None,
    breakeven: Optional[float] = None,
) -> BacktestConfig:
    return BacktestConfig(
        leverage=leverage,
        taker_fee=settings.taker_fee,
        slippage_bps=settings.slippage_bps,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        trail_pct=trail,
        breakeven_at_pct=breakeven,
    )


def _table(df: pd.DataFrame, title: str) -> Table:
    t = Table(title=title, title_justify="left")
    for col in df.columns:
        t.add_column(str(col), justify="left" if df[col].dtype == object else "right")
    for row in df.itertuples(index=False):
        t.add_row(*[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
    return t


def _metrics_table(metrics: dict, title: str) -> Table:
    t = Table(title=title, title_justify="left", show_header=False)
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k, v in metrics.items():
        t.add_row(k.replace("_", " "), f"{v:,}" if isinstance(v, (int, float)) else str(v))
    return t


def _stack(settings: Settings, dry_run: bool = False, live: bool = False):
    """Broker + risk manager + executor, with the mainnet safety check."""
    from .exchange import make_broker
    from .risk import Executor, Journal, RiskManager

    if settings.mode == "mainnet" and not dry_run and not (live or settings.confirm_mainnet):
        console.print(
            "[red]HL_MODE=mainnet trades real money.[/red] Re-run with --live "
            "(or set HL_CONFIRM_MAINNET=yes for unattended runs), or use --dry-run."
        )
        raise typer.Exit(2)
    broker = make_broker(settings)
    executor = Executor(settings, broker, RiskManager(settings), Journal(settings.state_dir), dry_run=dry_run)
    return broker, executor


def _banner(settings: Settings, dry_run: bool) -> None:
    color = {"paper": "green", "testnet": "yellow", "mainnet": "red"}[settings.mode]
    label = settings.mode.upper() + (" + DRY RUN" if dry_run else "")
    console.print(f"[bold {color}]mode: {label}[/bold {color}]  coins: {', '.join(settings.coins)}")


# ---------------------------------------------------------------- research
@app.command()
def strategies():
    """List the built-in strategies and their parameters."""
    t = Table(title="Strategies", title_justify="left")
    t.add_column("name")
    t.add_column("what it does")
    t.add_column("defaults")
    for name, cls in REGISTRY.items():
        t.add_row(name, cls.description, ", ".join(f"{k}={v}" for k, v in cls.defaults.items()))
    console.print(t)


@app.command()
def fetch(coin: str, interval: str = O_INTERVAL, days: float = O_DAYS, source: str = O_SOURCE):
    """Download and cache candles and funding for a coin."""
    s = load_settings()
    df = _load(s, coin.upper(), interval, days, source, None)
    console.print(f"{len(df)} bars, {df.index[0]} to {df.index[-1]}, cached in {s.data_dir}/")


@app.command()
def backtest(
    coin: str,
    strategy: str = typer.Option("ema_cross", "--strategy", "-s"),
    interval: str = O_INTERVAL,
    days: float = O_DAYS,
    source: str = O_SOURCE,
    csv: Optional[str] = O_CSV,
    leverage: float = O_LEVERAGE,
    param: Optional[list[str]] = O_PARAM,
    stop_loss: Optional[float] = O_SL,
    take_profit: Optional[float] = O_TP,
    trail: Optional[float] = typer.Option(None, "--trail", help="Trailing stop distance in percent"),
    breakeven: Optional[float] = typer.Option(None, "--breakeven", help="Move the stop to entry after this profit in percent"),
    report: bool = typer.Option(True, help="Write an HTML report and a trades CSV to reports/"),
):
    """Backtest one strategy on one coin."""
    from .report import write_report

    s = load_settings()
    coin = coin.upper()
    df = _load(s, coin, interval, days, source, csv)
    strat = get_strategy(strategy, **_params(param))
    res = run_backtest(df, strat, _cfg(s, leverage, stop_loss, take_profit, trail, breakeven))
    console.print(_metrics_table(res.metrics, f"{strat!r} on {coin} {interval}, {len(df)} bars"))
    if report:
        base = s.reports_dir / f"{coin}_{interval}_{strategy}"
        path = write_report(res, df["close"], coin, interval, base.with_suffix(".html"))
        res.trades.to_csv(base.with_name(base.name + "_trades.csv"), index=False)
        console.print(f"Report: {path}")


@app.command()
def compare(
    coin: str,
    interval: str = O_INTERVAL,
    days: float = O_DAYS,
    source: str = O_SOURCE,
    csv: Optional[str] = O_CSV,
    leverage: float = O_LEVERAGE,
    stop_loss: Optional[float] = O_SL,
):
    """Backtest every strategy with default parameters and rank them."""
    from .signals import rank_strategies

    s = load_settings()
    df = _load(s, coin.upper(), interval, days, source, csv)
    ranked = rank_strategies(df, _cfg(s, leverage, stop_loss, None))
    console.print(_table(ranked, f"{coin.upper()} {interval}, {len(df)} bars, {leverage}x, costs included"))


@app.command()
def optimize(
    coin: str,
    strategy: str = typer.Option(..., "--strategy", "-s"),
    interval: str = O_INTERVAL,
    days: float = O_DAYS,
    source: str = O_SOURCE,
    csv: Optional[str] = O_CSV,
    leverage: float = O_LEVERAGE,
    metric: str = typer.Option("sharpe", help="Column to rank by"),
    top: int = typer.Option(10, help="Rows to show"),
):
    """Grid-search a strategy's parameters. The best row is fitted to the past: confirm with walkforward."""
    s = load_settings()
    df = _load(s, coin.upper(), interval, days, source, csv)
    with console.status("Running the parameter grid..."):
        ranked = grid_search(df, strategy, _cfg(s, leverage, None, None), metric=metric)
    keys = list(get_strategy(strategy).grid)
    cols = keys + ["total_return_pct", "sharpe", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor"]
    console.print(_table(ranked[cols].head(top), f"{strategy} on {coin.upper()} {interval}: top {top} of {len(ranked)} by {metric}"))
    console.print("[yellow]These parameters were picked with hindsight. Run `hyperbot walkforward` for an out-of-sample check.[/yellow]")


@app.command()
def walkforward(
    coin: str,
    strategy: str = typer.Option(..., "--strategy", "-s"),
    interval: str = O_INTERVAL,
    days: float = O_DAYS,
    source: str = O_SOURCE,
    csv: Optional[str] = O_CSV,
    leverage: float = O_LEVERAGE,
    splits: int = typer.Option(4, help="Number of out-of-sample folds"),
    metric: str = typer.Option("sharpe"),
):
    """Walk-forward test: tune on the past, trade the next unseen slice, repeat."""
    s = load_settings()
    df = _load(s, coin.upper(), interval, days, source, csv)
    with console.status("Running walk-forward folds..."):
        wf = walk_forward(df, strategy, _cfg(s, leverage, None, None), n_splits=splits, metric=metric)
    folds = pd.DataFrame(wf["folds"])
    folds["params"] = folds["params"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
    console.print(_table(folds, f"{strategy} on {coin.upper()} {interval}: walk-forward folds"))
    console.print(_metrics_table(wf["oos_metrics"], "Stitched out-of-sample result"))
    console.print(
        f"Average {metric}: in-sample {wf['in_sample_avg']}, out-of-sample {wf['out_of_sample_avg']}. "
        "A large drop out-of-sample means the parameters were overfitted."
    )


@app.command()
def signals(coin: str, interval: str = O_INTERVAL):
    """Current indicators and what every strategy says for a coin."""
    from .signals import quant_snapshot

    s = load_settings()
    df = _load(s, coin.upper(), interval, 600 * pd.Timedelta(interval).total_seconds() / 86400, "hyperliquid", None)
    snap = quant_snapshot(df)
    console.print(_metrics_table(snap["indicators"], f"{coin.upper()} {interval} as of {snap['as_of_bar_close']}"))
    console.print(_metrics_table({**snap["strategy_signals"], "consensus": snap["consensus"]}, "Strategy signals (+1 long, -1 short)"))


@app.command()
def sentiment(as_json: bool = typer.Option(False, "--json", help="Print raw JSON")):
    """Fear & Greed, news headlines and Hyperliquid positioning."""
    from .sentiment import snapshot

    s = load_settings()
    with console.status("Collecting sentiment..."):
        snap = snapshot(s.coins)
    if as_json:
        console.print_json(json.dumps(snap, default=str))
        return
    fg = snap.get("fear_greed")
    if fg:
        console.print(f"Fear & Greed: [bold]{fg['value']}[/bold] ({fg['label']}), {fg['change_7d']:+d} over 7 days")
    if snap.get("positioning"):
        console.print(_table(pd.DataFrame(snap["positioning"]).T.reset_index(names="coin"), "Hyperliquid positioning"))
    heads = snap.get("headlines", [])[:15]
    if heads:
        t = Table(title=f"Headlines (keyword score {snap.get('news_keyword_score')})", title_justify="left")
        t.add_column("score", justify="right")
        t.add_column("source")
        t.add_column("headline")
        for h in heads:
            t.add_row(f"{h['keyword_score']:+d}", h["source"], h["title"])
        console.print(t)
    console.print(f"Composite score: {snap.get('composite_score')}  (-1 bearish, +1 bullish; rough)")
    for err in snap.get("errors", []):
        console.print(f"[yellow]unavailable: {err}[/yellow]")


# ---------------------------------------------------------------- trading
@app.command()
def status():
    """Account, positions and risk state."""
    s = load_settings()
    _banner(s, False)
    broker, executor = _stack(s, dry_run=True)
    acct = executor.account()
    d = acct.to_dict()
    positions = d.pop("positions")
    console.print(_metrics_table(d, "Account"))
    if positions:
        console.print(_table(pd.DataFrame(positions), "Positions"))
    pnl = executor.pnl_report(acct.equity)
    pnl.pop("note")
    console.print(_metrics_table(pnl, "Profit and loss (net = trading PnL minus AI cost)"))
    risk = executor.risk.status(acct.equity)
    limits = risk.pop("limits")
    console.print(_metrics_table(risk, "Risk state"))
    console.print(_metrics_table(limits, "Hard limits"))


@app.command()
def trade(
    coin: str,
    target_usd: float = typer.Argument(..., help="Signed position value: 500 = long $500, -500 = short, 0 = flat"),
    stop_loss: Optional[float] = O_SL,
    take_profit: Optional[float] = O_TP,
    live: bool = O_LIVE,
    dry_run: bool = O_DRY,
):
    """Manually set a coin's position (goes through the same risk checks as the agent)."""
    s = load_settings()
    _banner(s, dry_run)
    _, executor = _stack(s, dry_run, live)
    console.print_json(json.dumps(executor.set_target(coin, target_usd, stop_loss, take_profit, source="cli"), default=str))


@app.command()
def close(
    coin: Optional[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Close every position"),
    live: bool = O_LIVE,
):
    """Close one position, or all of them."""
    s = load_settings()
    _banner(s, False)
    _, executor = _stack(s, False, live)
    if all_:
        results = executor.flatten_all(source="cli", reason="close --all")
    elif coin:
        results = [executor.set_target(coin, 0.0, source="cli", reason="close")]
    else:
        raise typer.BadParameter("Give a coin or --all")
    console.print_json(json.dumps(results, default=str))


@app.command()
def kill(live: bool = O_LIVE):
    """Emergency stop: close everything and block new positions until `resume`."""
    s = load_settings()
    _banner(s, False)
    _, executor = _stack(s, False, live)
    executor.risk.halt("manual kill switch")
    console.print_json(json.dumps(executor.flatten_all(source="cli", reason="kill switch"), default=str))
    console.print("[red]Trading halted.[/red] Run `hyperbot resume` to allow new positions again.")


@app.command()
def resume():
    """Clear a halt (manual kill or drawdown limit) and restart drawdown tracking."""
    s = load_settings()
    _, executor = _stack(s, dry_run=True)
    executor.risk.resume(executor.broker.account().equity)
    console.print("Trading resumed.")


@app.command("paper-reset")
def paper_reset():
    """Reset the paper account to its starting balance."""
    from .exchange import PaperBroker
    from .risk import RiskManager

    s = load_settings()
    PaperBroker(s).reset()
    rm = RiskManager(s)
    rm.state["baseline_equity"] = 0.0  # profit and loss start again from the new balance
    rm.resume(s.paper_equity)
    (s.state_dir / "llm_usage.json").unlink(missing_ok=True)
    console.print(f"Paper account reset to ${s.paper_equity:,.0f}.")


@app.command()
def journal(count: int = typer.Option(15, "--count", "-n")):
    """Show recent decisions and trades."""
    from .risk import Journal

    s = load_settings()
    for rec in Journal(s.state_dir).tail(count):
        body = rec.get("report") or json.dumps({k: v for k, v in rec.items() if k not in ("time", "kind")}, default=str)
        console.print(Panel(body, title=f"{rec['time']}  {rec['kind']}", title_align="left"))


@app.command()
def run(
    coin: str,
    strategy: str = typer.Option(..., "--strategy", "-s"),
    interval: str = O_INTERVAL,
    leverage: float = O_LEVERAGE,
    param: Optional[list[str]] = O_PARAM,
    stop_loss: Optional[float] = O_SL,
    take_profit: Optional[float] = O_TP,
    once: bool = typer.Option(False, "--once", help="Run a single step and exit"),
    live: bool = O_LIVE,
    dry_run: bool = O_DRY,
):
    """Trade one rule-based strategy on one coin (no LLM). Acts on each closed bar."""
    from .data import interval_ms
    from .runner import StrategyRunner, run_forever

    s = load_settings()
    coin = coin.upper()
    if coin not in s.coins:
        s = replace(s, coins=[*s.coins, coin])
    _banner(s, dry_run)
    _, executor = _stack(s, dry_run, live)
    runner = StrategyRunner(s, executor, get_strategy(strategy, **_params(param)), coin, interval, leverage, stop_loss, take_profit)

    def step():
        console.log(json.dumps(runner.step(), default=str))

    if once:
        step()
    else:
        run_forever(step, interval_ms(interval) // 1000, console.log, align=True)


@app.command()
def agent(
    once: bool = typer.Option(False, "--once", help="Run one decision cycle and exit"),
    every: str = typer.Option("1h", "--every", "-e", help="Time between cycles: 15m, 1h, 4h"),
    live: bool = O_LIVE,
    dry_run: bool = O_DRY,
):
    """Run the LLM agent: it reads quant signals and sentiment, then manages positions."""
    from .agent import AgentError, TradingAgent
    from .runner import parse_duration, run_forever

    s = load_settings()
    _banner(s, dry_run)
    console.print(f"model: {s.agent_model} at {s.llm_base_url}")
    _, executor = _stack(s, dry_run, live)

    def on_event(kind: str, text: str) -> None:
        if kind == "tool":
            console.log(f"[cyan]tool[/cyan] {text[:300]}")
        elif kind == "action":
            console.log(f"[magenta]trade result[/magenta] {text[:600]}")

    try:
        bot = TradingAgent(s, executor, HLData(), on_event=on_event)
    except AgentError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)

    def cycle():
        summary = bot.run_cycle()
        console.print(Panel(summary["report"], title="Agent report", title_align="left"))
        u = summary["usage"]
        console.log(f"{len(summary['actions'])} trade action(s); tokens in {u['prompt_tokens']:,} / out {u['completion_tokens']:,}; AI cost ${summary['cost_usd']:.4f}")

    if once:
        try:
            cycle()
        except AgentError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        run_forever(cycle, parse_duration(every), console.log)


def _show_scorecard(card: dict, top: int) -> None:
    df = pd.DataFrame(card["rows"])
    if df.empty:
        console.print("[yellow]The scorecard is empty.[/yellow]")
        return
    df["live_params"] = df["live_params"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
    df["folds+"] = df["oos_folds_positive"].astype(str) + "/" + df["oos_folds"].astype(str)
    cols = ["coin", "interval", "strategy", "oos_return_pct", "oos_sharpe", "psr", "dsr", "half_kelly", "oos_max_dd_pct", "folds+",
            "oos_trades", "fitted_return_pct", "current_signal", "live_params"]
    cols = [c for c in cols if c in df.columns]
    console.print(_table(df[cols].head(top), f"Research scorecard ({card['generated']}, {card['days']:g} days): top {min(top, len(df))} of {len(df)} by out-of-sample Sharpe"))
    console.print(
        "oos = walk-forward on unseen data (the honest number); fitted = best parameters picked with hindsight.\n"
        "psr = chance the true Sharpe is above zero; dsr = the same after allowing for how many candidates were tested; "
        "half_kelly = leverage ceiling for that strategy."
    )


def _show_playbook(pb: Optional[dict]) -> None:
    if not pb:
        console.print("No playbook yet. `hyperbot auto` creates one.")
        return
    console.print(f"Playbook from [bold]{pb['source']}[/bold], updated {pb['updated']}, risk scale {pb['risk_scale']}")
    if pb["allocations"]:
        df = pd.DataFrame(pb["allocations"])
        df["params"] = df["params"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
        console.print(_table(df, "Allocations (weight = share of equity at full signal)"))
    else:
        console.print("[yellow]No allocations: the account stays flat.[/yellow]")
    if pb.get("rationale"):
        console.print(Panel(pb["rationale"], title="Rationale", title_align="left"))


@app.command()
def research(
    refresh: bool = typer.Option(False, "--refresh", help="Re-run even if a recent scorecard exists"),
    top: int = typer.Option(25, help="Rows to show"),
):
    """Backtest and walk-forward every strategy on every coin; the autopilot chooses from this."""
    from .auto import build_scorecard, load_scorecard

    s = load_settings()
    card = None if refresh else load_scorecard(s)
    if card is None:
        console.print("Running research on every coin, timeframe and strategy. This takes a few minutes.")
        card = build_scorecard(s, console.log)
    _show_scorecard(card, top)


@app.command()
def playbook():
    """Show the strategies the autopilot is currently running."""
    from .auto import load_playbook

    _show_playbook(load_playbook(load_settings()))


@app.command()
def auto(
    tick: str = typer.Option("5m", "--tick", help="How often signals are checked and positions adjusted"),
    review_every: str = typer.Option("4h", "--review-every", help="How often the LLM reviews the playbook"),
    research_every: str = typer.Option("24h", "--research-every", help="How often the backtests are re-run"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Pick strategies by a fixed rule instead of the LLM"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    live: bool = O_LIVE,
    dry_run: bool = O_DRY,
):
    """Fully automatic: research, let the AI pick strategies, trade them, repeat."""
    from .auto import AutoPilot, load_playbook
    from .runner import parse_duration, run_forever

    s = load_settings()
    _banner(s, dry_run)
    _, executor = _stack(s, dry_run, live)
    use_llm = not no_llm
    if use_llm and not s.llm_api_key:
        console.print("[yellow]DEEPSEEK_API_KEY is not set: strategies will be picked by the built-in rule, not the AI.[/yellow]")
        use_llm = False

    def on_event(kind: str, text: str) -> None:
        if kind in ("tool", "action"):
            console.log(f"[cyan]{kind}[/cyan] {text[:300]}")

    pilot = AutoPilot(s, executor, use_llm, parse_duration(review_every), parse_duration(research_every), console.log, on_event)

    def step():
        results = pilot.tick()
        acct = executor.account()
        moved = [r for r in results if r["status"] not in ("hold", "noop")]
        pnl = executor.pnl_report(acct.equity)
        console.log(
            f"equity ${acct.equity:,.2f}, net PnL after AI cost ${pnl['net_pnl_usd']:+,.2f} (AI ${pnl['ai_cost_usd']:.2f}), {len(acct.positions)} position(s), "
            + (", ".join(f"{r['coin']} {r['target_fraction']:+.2f} {r['status']}" for r in moved) or "no changes")
        )

    if once:
        step()
        _show_playbook(load_playbook(s))
    else:
        run_forever(step, parse_duration(tick), console.log)


@app.command()
def simulate(
    amount: float = typer.Option(10_000.0, "--amount", "-a", help="Starting balance in USD"),
    name: str = typer.Option("default", "--name", help="Run several simulations side by side under different names"),
    tick: str = typer.Option("5m", "--tick"),
    review_every: str = typer.Option("4h", "--review-every"),
    research_every: str = typer.Option("24h", "--research-every"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Pick strategies by the fixed rule instead of the LLM"),
    once: bool = typer.Option(False, "--once", help="Run a single tick and exit"),
    report: bool = typer.Option(False, "--report", help="Only show how the simulation has done so far"),
    reset: bool = typer.Option(False, "--reset", help="Delete this simulation and start again"),
):
    """Run the autopilot on pretend money against the live market, with realistic fills, and track the result."""
    import shutil

    from .auto import AutoPilot
    from .exchange import PaperBroker
    from .risk import AICostLedger, Executor, Journal, RiskManager
    from .runner import parse_duration, run_forever
    from .sim import SimTracker

    base = load_settings()
    sim_dir = base.state_dir / "sim" / name
    if reset and sim_dir.is_dir():
        shutil.rmtree(sim_dir)
        console.print(f"Simulation '{name}' deleted.")
    s = replace(base, mode="paper", state_dir=sim_dir, paper_equity=amount)
    s.ensure_dirs()
    broker = PaperBroker(s)
    tracker = SimTracker(s, broker)

    def show_report():
        rep = tracker.report()
        if rep:
            console.print(_metrics_table(rep, f"Simulation '{name}'"))
        else:
            console.print(f"Simulation '{name}' has not recorded anything yet.")

    if report:
        show_report()
        return
    started = broker.state.get("start_equity", amount)
    if started != amount:
        console.print(f"[yellow]Simulation '{name}' already exists with ${started:,.0f}; continuing it. Use --reset to start over with ${amount:,.0f}.[/yellow]")
    console.print(f"[bold green]SIMULATION '{name}'[/bold green]: ${started:,.0f} of pretend money on live prices. Coins: {', '.join(s.coins)}")
    executor = Executor(s, broker, RiskManager(s), Journal(s.state_dir))
    use_llm = not no_llm
    if use_llm and not s.llm_api_key:
        console.print("[yellow]DEEPSEEK_API_KEY is not set: strategies will be picked by the built-in rule, not the AI.[/yellow]")
        use_llm = False

    def on_event(kind: str, text: str) -> None:
        if kind in ("tool", "action"):
            console.log(f"[cyan]{kind}[/cyan] {text[:300]}")

    pilot = AutoPilot(s, executor, use_llm, parse_duration(review_every), parse_duration(research_every), console.log, on_event)
    ledger = AICostLedger(s)

    def step():
        results = pilot.tick()
        acct = executor.account()
        row = tracker.record(acct, broker.mids())
        moved = [r for r in results if r["status"] not in ("hold", "noop")]
        console.log(
            f"equity ${row['equity']:,.2f} ({(row['equity'] / started - 1) * 100:+.2f}%), net of AI cost ${row['equity'] - started - ledger.total():+,.2f}, "
            f"buy-and-hold ${row['benchmark']:,.2f}, {len(acct.positions)} position(s), "
            + (", ".join(f"{r['coin']} {r['target_fraction']:+.2f} {r['status']}" for r in moved) or "no changes")
        )

    if once:
        step()
        show_report()
    else:
        run_forever(step, parse_duration(tick), console.log)
        show_report()


@app.command()
def replay(
    amount: float = typer.Option(10_000.0, "--amount", "-a", help="Starting balance in USD"),
    days: float = typer.Option(150.0, "--days", "-d", help="How far back to replay"),
    interval: str = typer.Option("4h", "--interval", "-i"),
    repick_days: int = typer.Option(14, "--repick-days", help="How often strategies are re-chosen"),
    lookback_days: float = typer.Option(150.0, "--lookback-days", help="History used for each research run"),
):
    """Replay the rule-based autopilot over past months: research, pick, trade, re-pick, with no hindsight."""
    from .sim import replay as run_replay

    s = load_settings()
    console.print(
        f"Replaying {days:g} days on {interval} candles with ${amount:,.0f}; strategies re-picked every {repick_days} days "
        "from earlier data only. This takes a few minutes."
    )
    out = run_replay(s, amount, days, interval, repick_days, lookback_days, console.log)
    console.print(_table(pd.DataFrame(out["periods"]), "Periods"))
    console.print(_metrics_table(out["metrics"], "Replay result (buy hold = equal-weight basket of the coins)"))
    path = s.reports_dir / f"replay_{interval}_{int(days)}d.csv"
    pd.DataFrame({"equity": out["equity"], "buy_and_hold": out["benchmark"]}).to_csv(path)
    console.print(f"Equity curve: {path}")
    console.print("[yellow]The LLM is not part of a replay: a language model may already know what happened in this period.[/yellow]")


@app.command()
def mcp():
    """Start the MCP server (stdio) so an AI client such as Claude Code can use the bot."""
    from .mcp_server import main

    main()


@app.command()
def api(
    host: str = typer.Option("127.0.0.1", "--host", help="Use 0.0.0.0 to accept remote connections (needs HL_API_TOKEN)"),
    port: int = typer.Option(8787, "--port"),
):
    """Start the HTTP API."""
    import os

    import uvicorn

    from .api import create_app

    token = os.getenv("HL_API_TOKEN", "").strip()
    load_settings()  # loads .env
    token = token or os.getenv("HL_API_TOKEN", "").strip()
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        console.print("[red]Refusing to listen on a public address without HL_API_TOKEN set.[/red]")
        raise typer.Exit(2)
    console.print(f"API on http://{host}:{port}  (docs at /docs, token {'required' if token else 'not set: localhost only'})")
    uvicorn.run(create_app(token=token), host=host, port=port, log_level="warning")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: Optional[int] = typer.Option(None, "--port", help="Defaults to $PORT, then 8787"),
    tick: str = typer.Option("5m", "--tick", envvar="HL_TICK"),
    review_every: str = typer.Option("4h", "--review-every", envvar="HL_REVIEW_EVERY"),
    research_every: str = typer.Option("24h", "--research-every", envvar="HL_RESEARCH_EVERY"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    live: bool = O_LIVE,
):
    """Autopilot and HTTP API in one process, sharing one account. Made for cloud hosts."""
    import os
    import threading
    import time

    import uvicorn

    from .api import create_app
    from .auto import AutoPilot
    from .runner import parse_duration
    from .service import Service

    s = load_settings()
    token = os.getenv("HL_API_TOKEN", "").strip()
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        console.print("[red]Refusing to listen on a public address without HL_API_TOKEN set.[/red]")
        raise typer.Exit(2)
    _banner(s, False)
    _, executor = _stack(s, False, live)
    use_llm = not no_llm and bool(s.llm_api_key)
    if not use_llm:
        console.print("Strategies are picked by the built-in rule (no LLM key, or --no-llm).")
    lock = threading.Lock()
    pilot = AutoPilot(s, executor, use_llm, parse_duration(review_every), parse_duration(research_every), console.log, lock=lock)
    service = Service(s, executor=executor, runner=pilot.runner, lock=lock)
    wait = parse_duration(tick)

    def loop():
        while True:
            try:
                results = pilot.tick()  # takes the lock itself, only around account reads and trades
                with lock:
                    acct = executor.account()
                    pnl = executor.pnl_report(acct.equity)
                moved = [r for r in results if r["status"] not in ("hold", "noop")]
                console.log(
                    f"equity ${acct.equity:,.2f}, net PnL after AI cost ${pnl['net_pnl_usd']:+,.2f}, {len(acct.positions)} position(s), "
                    + (", ".join(f"{r['coin']} {r['target_fraction']:+.2f} {r['status']}" for r in moved) or "no changes")
                )
            except Exception as e:  # keep the API up and try again next tick
                console.log(f"autopilot tick failed ({type(e).__name__}): {e}")
            time.sleep(wait)

    threading.Thread(target=loop, daemon=True, name="autopilot").start()
    port = port or int(os.getenv("PORT", "8787"))
    console.print(f"API on http://{host}:{port} (token {'required' if token else 'not set: localhost only'}); autopilot ticking every {tick}")
    uvicorn.run(create_app(service, token=token), host=host, port=port, log_level="warning")


@app.command()
def setup():
    """Interactive first-time setup: asks for your keys and writes .env."""
    from pathlib import Path

    env = Path(".env")
    template = env if env.is_file() else Path(".env.example")
    lines = template.read_text(encoding="utf-8").splitlines() if template.is_file() else []
    console.print("Keys are written to .env in this folder and nowhere else. Press Enter to keep a current value.")
    mode = typer.prompt("Mode (paper / testnet / mainnet)", default="paper").strip().lower()
    if mode not in ("paper", "testnet", "mainnet"):
        raise typer.BadParameter("mode must be paper, testnet or mainnet")
    values = {"HL_MODE": mode}
    if mode != "paper":
        key = typer.prompt("Hyperliquid API wallet private key (hidden)", default="", hide_input=True, show_default=False).strip()
        if key:
            values["HL_PRIVATE_KEY"] = key
        addr = typer.prompt("Main account address (blank if the key is the main wallet itself)", default="", show_default=False).strip()
        if addr:
            values["HL_ACCOUNT_ADDRESS"] = addr
    if mode == "mainnet":
        ok = typer.confirm("Allow unattended real-money trading (sets HL_CONFIRM_MAINNET=yes)?", default=False)
        values["HL_CONFIRM_MAINNET"] = "yes" if ok else "no"
    llm = typer.prompt("DeepSeek API key (hidden, blank to pick strategies by rule)", default="", hide_input=True, show_default=False).strip()
    if llm:
        values["DEEPSEEK_API_KEY"] = llm
    values["HL_COINS"] = typer.prompt("Coins to trade", default="BTC,ETH,SOL").upper().replace(" ", "")
    out, seen = [], set()
    for line in lines:
        name = line.split("=", 1)[0].strip()
        if name in values and not line.lstrip().startswith("#"):
            out.append(f"{name}={values[name]}")
            seen.add(name)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in seen]
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    console.print("Saved .env. Next: `hyperbot doctor`, then `hyperbot auto`.")


@app.command()
def doctor():
    """Check configuration and connectivity without trading."""
    s = load_settings()
    _banner(s, False)
    rows = []

    def check(name: str, fn):
        try:
            rows.append((name, "ok", str(fn())))
        except Exception as e:
            rows.append((name, "FAIL", f"{type(e).__name__}: {e}"))

    check("Hyperliquid market data", lambda: f"BTC mid {HLData().mids()['BTC']:,.0f}")
    check("LLM API key", lambda: "set" if s.llm_api_key else (_ for _ in ()).throw(RuntimeError("DEEPSEEK_API_KEY is not set (needed only for `agent`)")))
    if s.is_live:
        def live_check():
            from .exchange import HyperliquidBroker

            b = HyperliquidBroker(s)
            return f"account {b.address[:6]}...{b.address[-4:]} ({b.abstraction}) equity ${b.account().equity:,.2f}"

        check(f"{s.mode} account", live_check)
    else:
        check("Paper account", lambda: f"equity ${_stack(s, dry_run=True)[1].account().equity:,.2f}")
    t = Table(title="Checks", title_justify="left")
    for col in ("check", "result", "detail"):
        t.add_column(col)
    for r in rows:
        t.add_row(*r)
    console.print(t)


if __name__ == "__main__":
    app()
