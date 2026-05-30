# Post-Earnings-Announcement Drift (PEAD) Dashboard

A self-updating dashboard that simulates a **post-earnings-announcement drift**
strategy over a curated watchlist and publishes the results as a static site.

> **🛠 Maintenance note:** This README documents how the model works. **Keep it
> up to date** — whenever the strategy, data pipeline, or dashboard changes
> (e.g. new constants, a different signal, a new output field), update the
> relevant section here in the same change.

## The idea

PEAD is a well-documented market anomaly: when a company beats earnings
expectations, its stock tends to keep *drifting* upward for weeks afterward
rather than repricing instantly. This project finds positive earnings
surprises, simulates buying the day after the report, holds for ~60 days, and
measures how much the stock drifts — both raw and market-adjusted.

It is a **simulation** for research/illustration: no real orders, no
slippage or fees, long-only on positive surprises.

## Components

| File | Role |
|------|------|
| `tickers.txt` | The watchlist — the universe of stocks tracked (one symbol per line; `#` comments allowed). |
| `fetch_data.py` | The engine. Pulls data from Yahoo Finance, simulates trades, writes output. |
| `trades_log.json` | The **persistent log** of every trade ever opened. The source of truth; grows over time. |
| `data.json` | The rendered snapshot the website reads (summary stats + trades + upcoming earnings). |
| `index.html` | The dashboard — a static, client-side page that fetches `data.json` and renders charts/tables. |
| `.github/workflows/update.yml` | The scheduler — runs the engine every 2 hours via GitHub Actions. |

## How a run works

Each run is **incremental** — it scans only for *new* earnings events and
appends them to the log, rather than rebuilding the full history every time.

1. **Scheduler fires** (`update.yml`, cron every 2h): a GitHub Actions runner
   installs dependencies, runs `python fetch_data.py`, then commits the updated
   `data.json` and `trades_log.json` back to the repo.

2. **Scan for new events** (`fetch_data.py`):
   - Load `trades_log.json` (every trade ever recorded).
   - Pull cheap earnings *metadata* for each watchlist ticker (dates + reported
     vs. estimated EPS). Used to surface upcoming earnings, find each stock's
     next-earnings date (for exits), and spot **new events** — recent earnings
     (within `NEW_EVENT_LOOKBACK_DAYS`) not already in the log.
   - Fetch **prices only for what must be (re)simulated**: trades still **open**
     (to refresh/finalize them) plus those new events, over a short recent
     window. Closed trades already in the log are never re-fetched.

3. **Turn surprises into trades** (`simulate_trade`):
   - **SUE** ("surprise yield") = EPS surprise ÷ share price at earnings, as a
     percent. This is the signal strength.
   - Only events with **SUE ≥ `MIN_SUE`** become long trades (PEAD is
     directional; negative/marginal surprises are skipped).
   - **Entry:** first trading day after earnings. **Exit:** `HOLD_DAYS` later,
     or the day before the *next* earnings report if that comes first.
   - Returns are computed both **raw** and **abnormal** (stock return minus the
     benchmark over the same window) — the abnormal figure is the one that
     reflects drift, with a t-stat on the abnormal returns.

4. **Upsert into the log:** new events are added; open trades are refreshed and
   flip to **closed** once their exit passes. Closed trades are immutable —
   recorded once and kept forever.

5. **Write `data.json`:** aggregate stats (win rate, avg raw & abnormal return,
   t-stat), the full trade list, upcoming earnings, and data-source status.

6. **Dashboard** (`index.html`): a static page that `fetch()`es `data.json` and
   renders the SUE-vs-return scatter, return distribution, returns-by-quarter,
   and the open/closed trade tables. No backend — GitHub Pages serves the files.

## Lifecycle of one trade

```
Earnings beat (SUE ≥ MIN_SUE)
   → buy next trading day                          [open]
   → refreshed every run for ~HOLD_DAYS            [open, returnPct updates]
   → HOLD_DAYS pass (or next earnings hits first)
   → finalized with exit price + return            [closed, immutable]
```

## Why the persistent log

Originally every run rebuilt a trailing 5-year window from scratch, so older
trades silently dropped off and the same work was redone every 2 hours. The log
**accumulates**: each run only appends genuinely new trades and updates the few
still open. A run with nothing open and no new earnings does essentially no
price fetching. Over time the dashboard shows the full history, not a rolling
window.

## Key parameters

All live near the top of `fetch_data.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `MIN_SUE` | `0.1` | Minimum SUE (EPS surprise as % of price) to enter a long. |
| `HOLD_DAYS` | `60` | Target holding period after entry. |
| `NEW_EVENT_LOOKBACK_DAYS` | `120` | How far back to look for *new* events each run. |
| `UPCOMING_DAYS` | `30` | Window for the "upcoming earnings" list. |
| `BENCHMARK` | `SPY` | Proxy used for abnormal (market-adjusted) returns. |
| `LOOKBACK_YEARS` | `5` | Only documents how the initial log was seeded. |

## Operational notes

- **Data source:** Yahoo Finance via `yfinance`. Yahoo rate-limits datacenter
  IPs, so the engine uses a `curl_cffi` browser-impersonation session and a
  *last-known-good / stale* safety net: if a run needs live prices but gets
  none, it preserves the prior data and flags it stale instead of blanking the
  dashboard.
- **Edit the watchlist:** change `tickers.txt` and the next run picks it up.
- **Run locally:** `pip install yfinance pandas tzdata curl_cffi lxml` then
  `python fetch_data.py` (note: Yahoo may block non-residential IPs).
