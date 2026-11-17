# hyperbot

A command-line trading bot for Hyperliquid perpetuals:

- **Backtesting** with fees, slippage, funding, leverage, stops and liquidation, plus parameter search and walk-forward testing.
- **Ten strategies** (trend, momentum, breakout, mean reversion, funding carry, regime switching, ensemble).
- **An LLM agent** (DeepSeek by default) that reads quant signals, recent strategy performance and market sentiment, then manages positions.
- **A risk engine in plain code** that every order passes through. The agent cannot exceed it.
- **Three modes**: `paper` (simulated fills at real prices, the default), `testnet`, `mainnet`.

Nothing here is a proven money-maker. The built-in strategies are standard textbook rules, and on recent data most of them lose after costs. Treat the bot as a framework for testing your own ideas, and expect to lose money you put behind an untested one.

## Install

Linux, macOS or Git Bash on Windows, Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env
hyperbot doctor
```

## Research commands (no keys needed)

```bash
hyperbot strategies                                   # list strategies and parameters
hyperbot compare BTC -i 1h -d 180                     # rank every strategy on the same data
hyperbot backtest ETH -s donchian_breakout -i 4h -d 365 --stop-loss 5
hyperbot backtest BTC -s ema_cross -p fast=10 -p slow=50 --leverage 2
hyperbot optimize BTC -s ema_cross -i 1h              # grid search (fitted to the past)
hyperbot walkforward BTC -s ema_cross -i 1h           # honest out-of-sample check
hyperbot signals SOL -i 4h                            # indicators + every strategy's current call
hyperbot sentiment                                    # Fear & Greed, headlines, funding, open interest
```

`backtest` writes an HTML report and a trades CSV to `reports/`.

Hyperliquid only serves about 5000 candles per interval (about 7 months of 1h bars). For longer tests use `--interval 4h`, `--source binance`, or `--csv yourfile.csv` (columns: time, open, high, low, close; volume and funding optional). Downloaded candles are cached in `data/` and extended on each run.

## Which keys you need

| What you want | Needed in `.env` |
|---|---|
| Backtests, research, paper trading | Nothing |
| The AI choosing strategies | `DEEPSEEK_API_KEY` |
| Trading on Hyperliquid testnet | `HL_MODE=testnet`, `HL_PRIVATE_KEY`, `HL_ACCOUNT_ADDRESS` |
| Trading real money | `HL_MODE=mainnet`, `HL_PRIVATE_KEY`, `HL_ACCOUNT_ADDRESS`, `HL_CONFIRM_MAINNET=yes` |

Hyperliquid has no API key or secret like a centralised exchange. Orders are signed with a wallet private key. Create an API wallet at app.hyperliquid.xyz (More → API) and use its key: it can trade for your account but cannot withdraw. `HL_ACCOUNT_ADDRESS` is your main wallet's public address (0x...), which is where the funds are.

Everything else in `.env.example` (risk limits, costs, coins) has a working default.

## Trading

Paper mode works out of the box:

```bash
hyperbot trade BTC 2000 --stop-loss 2.5 --take-profit 5   # long $2000 (negative = short, 0 = flat)
hyperbot status
hyperbot run BTC -s donchian_breakout -i 4h               # rule-based strategy loop, no LLM
hyperbot close --all
hyperbot kill          # emergency stop: flatten and block new positions
hyperbot resume
hyperbot journal       # recent decisions and trades
hyperbot paper-reset
```

### Autopilot (fully automatic)

```bash
hyperbot setup      # asks for your keys once and writes .env
hyperbot auto       # research, pick strategies, trade, repeat
```

`auto` runs three jobs on a loop:

1. **Research** (every 24h): backtests and walk-forward tests every strategy on every coin and timeframe, and saves a scorecard. `hyperbot research` shows it.
2. **Choose** (every 4h, and after each research run): the LLM reads the scorecard, the market and sentiment, and writes the playbook: which strategies run, on which coins, at what weight. `hyperbot playbook` shows it. With `--no-llm`, or if the LLM is unreachable and no playbook exists yet, a fixed rule picks the strategies that held up best out-of-sample.
3. **Trade** (every 5 minutes): recomputes the playbook's signals and moves positions to match, through the risk engine. A stopped-out position is not re-entered until the strategies change side.

Options: `--tick 5m`, `--review-every 4h`, `--research-every 24h`, `--no-llm`, `--once`, `--dry-run`.

### The discretionary agent

A second way to use the LLM: instead of choosing strategies, it sets positions directly each cycle.


Set `DEEPSEEK_API_KEY` in `.env`, then:

```bash
hyperbot agent --once --dry-run     # one cycle, decides but places no orders
hyperbot agent --once               # one cycle on the paper account
hyperbot agent --every 1h           # keep running
```

Each cycle the model calls tools to read the account, market overview, quant signals on several timeframes, a ranking of which strategies have recently worked, and sentiment. It then sets target positions with stops and writes a report to `state/journal.jsonl`, which is its memory for the next cycle.

Any OpenAI-compatible API works: change `HL_LLM_BASE_URL`, `HL_AGENT_MODEL` and `HL_LLM_API_KEY`.

### Going live

1. On app.hyperliquid.xyz create an **API wallet** (More → API). Its key can trade but cannot withdraw. Do not use your main wallet's key.
2. In `.env` set `HL_PRIVATE_KEY` to the API wallet key and `HL_ACCOUNT_ADDRESS` to your main account address.
3. Set `HL_MODE=testnet` first and run for a while. The live order path has only been checked read-only here, so testnet is where you confirm fills, stops and leverage behave as expected.
4. For real money set `HL_MODE=mainnet` and pass `--live` (or `HL_CONFIRM_MAINNET=yes` for Docker).

The private key is read by the bot process only. The LLM never sees it: the model can only call the tools in `hyperbot/agent.py`.

## Risk limits

Set in `.env`, enforced in `hyperbot/risk.py` for the agent, the strategy runner and manual trades alike:

| Setting | Default | Meaning |
|---|---|---|
| `HL_COINS` | BTC,ETH,SOL | Only these can be opened |
| `HL_MAX_LEVERAGE` | 3 | Total position value / equity |
| `HL_MAX_POSITION_PCT` | 100 | One coin's position as % of equity |
| `HL_MAX_DAILY_LOSS_PCT` | 5 | Past this, only reductions are allowed until the next UTC day |
| `HL_MAX_DRAWDOWN_PCT` | 20 | Past this, trading halts until `hyperbot resume` |
| `HL_MAX_TRADES_PER_DAY` | 20 | Cap on position-increasing trades |
| `HL_DEFAULT_STOP_LOSS_PCT` / `HL_MAX_STOP_LOSS_PCT` | 3 / 10 | Every position gets a stop |

Orders that reduce risk are always allowed. Requests above a cap are shrunk to fit.

## Docker

```bash
cp .env.example .env      # fill in keys
docker compose up -d --build
docker compose logs -f bot
docker compose run --rm bot compare BTC -i 4h -d 365    # any CLI command
docker compose down
```

State, cached data and reports are kept in `./state`, `./data` and `./reports` on the host. On a cloud VM, copy the folder, install Docker and run the same commands.

## Known limits

- Backtests fill at the next bar's open with fixed slippage. They ignore order-book depth, partial fills and latency.
- Paper mode checks stops only when the bot polls, and does not charge funding.
- News sentiment is a keyword count plus the model's reading of headlines. Headlines are untrusted text; the risk limits are the real protection against a manipulated feed.
- The risk engine sees only this bot's account. If you trade the same account by hand, its counters will still apply.

## Layout

```
hyperbot/
  data.py        market data and caching        strategies.py  the strategies
  indicators.py  indicators                     backtest.py    simulator, grid search, walk-forward
  signals.py     current quant snapshot         sentiment.py   Fear & Greed, news, positioning
  exchange.py    paper and live brokers         risk.py        limits, executor, journal
  agent.py       LLM agent and its tools        runner.py      loops and the strategy runner
  auto.py        autopilot: research, playbook, portfolio execution
  report.py      HTML reports                   cli.py         commands
tests/           pytest suite (python -m pytest)
```
