#!/usr/bin/env python3
"""
Fetch earnings surprise data + historical prices for a curated watchlist via
yfinance. Simulates a PEAD strategy and outputs per-trade data + summary stats
to data.json.

Methodology notes (see README / PR for rationale):
  * Universe: a hand-curated watchlist in tickers.txt (one symbol per line,
    '#' comments allowed). Edit that file to change which stocks are tracked.
  * Each earnings event becomes a long trade only when its SUE clears a
    positive threshold (MIN_SUE); negative/marginal-SUE events are skipped.
    PEAD is directional, so trading non-positive surprises long just fights
    the drift.
  * SUE is the EPS surprise scaled by the stock's price at the earnings date
    (a unitless "surprise yield"), used both to gate entry and for display.
  * Returns are reported both raw and market-adjusted (abnormal = stock - SPY
    over the same holding window), with a t-stat on the abnormal returns.
"""

import json
import math
import bisect
import datetime
import time
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

# Yahoo increasingly serves empty/4xx responses to datacenter IPs (e.g. CI
# runners), which makes every ticker look "delisted" and zeroes out the run.
# A browser-impersonating curl_cffi session gets past most of that; fall back
# to yfinance's default session if curl_cffi isn't installed.
try:
    from curl_cffi import requests as _cffi_requests
    _SESSION = _cffi_requests.Session(impersonate="chrome")
    print("Using curl_cffi browser-impersonation session")
except Exception as _e:  # curl_cffi missing or failed to init
    _SESSION = None
    print(f"curl_cffi unavailable ({_e}); using yfinance default session")


def _ticker(sym):
    # Defensive: some yfinance versions don't accept a custom session arg.
    if _SESSION is not None:
        try:
            return yf.Ticker(sym, session=_SESSION)
        except TypeError:
            pass
    return yf.Ticker(sym)


def _retry(fn, label, tries=3, base=1.5):
    """Call fn() with retries + exponential backoff.

    Treats a None/empty result as retryable (Yahoo often returns an empty body
    on a soft rate-limit). Returns the last result — possibly empty — once the
    retries are exhausted, so callers keep their existing empty-handling."""
    out = None
    for i in range(tries):
        try:
            out = fn()
            if out is not None and (not hasattr(out, '__len__') or len(out)):
                return out
        except Exception as e:
            print(f"({label} retry {i + 1} err: {e})", end=" ", flush=True)
        if i < tries - 1:
            time.sleep(base * (2 ** i))
    return out


LOOKBACK_YEARS = 5
HOLD_DAYS = 60
UPCOMING_DAYS = 30

# Minimum SUE (EPS surprise as a % of share price) required to enter a long.
# PEAD is directional — positive surprises drift up — so we only trade events
# whose SUE clears this positive threshold and skip everything below it.
MIN_SUE = 0.1

BENCHMARK = "SPY"        # broad-market proxy for abnormal-return calculation

# Curated, user-editable watchlist. The dashboard's "Watchlist" chip links here.
TICKERS_FILE = "tickers.txt"
WATCHLIST_URL = (
    "https://github.com/hansenvalueinvesting/post-earnings-announcement-drift/"
    "blob/main/tickers.txt"
)

# Built-in fallback used only if tickers.txt is missing/empty.
DEFAULT_TICKERS = [
    "NVDA", "MSFT", "GOOG", "META", "AMZN", "SPOT", "NFLX",
    "AXP", "V", "MA", "COST", "MCD", "KO", "MNST",
]


def load_tickers():
    """Read the watchlist from tickers.txt.

    Symbols may be separated by newlines, commas, or spaces; '#' starts a
    comment. Dots are normalized to dashes to match yfinance (BRK.B -> BRK-B).
    Falls back to DEFAULT_TICKERS if the file is missing or empty."""
    try:
        with open(TICKERS_FILE) as f:
            text = f.read()
    except FileNotFoundError:
        print(f"{TICKERS_FILE} not found — using built-in default watchlist")
        text = ""

    tickers, seen = [], set()
    for line in text.splitlines():
        line = line.split('#', 1)[0]
        for tok in line.replace(',', ' ').split():
            t = tok.strip().upper().replace('.', '-')
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)

    if not tickers:
        tickers = list(DEFAULT_TICKERS)
    print(f"Loaded {len(tickers)} watchlist tickers from {TICKERS_FILE}")
    return tickers


def fetch_earnings(sym, today_str):
    """Fetch earnings dates. Returns confirmed history (with EPS) and upcoming dates."""
    try:
        tk = _ticker(sym)
        df = _retry(lambda: tk.get_earnings_dates(limit=28), f"{sym} earnings")
        if df is None or df.empty:
            return {'history': [], 'upcoming': []}

        history, upcoming = [], []
        for idx, row in df.iterrows():
            dt = idx
            if hasattr(dt, 'date'):
                dt = dt.date()
            ds = str(dt)[:10]
            if ds > today_str:
                upcoming.append(ds)
                continue
            est = row.get('EPS Estimate')
            act = row.get('Reported EPS')
            if pd.isna(est) or pd.isna(act):
                continue
            history.append({
                'date': ds,
                'actual': float(act),
                'estimate': float(est),
                'surprise': round(float(act) - float(est), 4)
            })
        history.sort(key=lambda x: x['date'])
        upcoming = sorted(set(upcoming))
        return {'history': history, 'upcoming': upcoming}
    except Exception:
        return {'history': [], 'upcoming': []}


def fetch_prices(sym, start):
    """Fetch daily close prices."""
    try:
        df = _retry(lambda: _ticker(sym).history(start=start, auto_adjust=True),
                    f"{sym} prices")
        if df is None or df.empty:
            return []
        return [{'date': d.strftime('%Y-%m-%d'), 'close': round(float(r['Close']), 2)}
                for d, r in df.iterrows()]
    except Exception:
        return []


def _build_index(px):
    """Build (dates, closes) parallel arrays for as-of price lookup."""
    return [p['date'] for p in px], [p['close'] for p in px]


def _price_asof(idx, date_str):
    """Most recent close on/before date_str, or None if date precedes all data."""
    dates, closes = idx
    i = bisect.bisect_right(dates, date_str) - 1
    return closes[i] if i >= 0 else None


def quarter_of(date_str):
    d = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def main():
    now = datetime.datetime.now(ZoneInfo('America/New_York'))
    today = now.date()
    today_str = str(today)
    updated_at = now.strftime('%Y-%m-%d %H:%M:%S ET')
    cutoff = today - datetime.timedelta(days=LOOKBACK_YEARS * 365)
    cutoff_str = str(cutoff)
    upcoming_cutoff = str(today + datetime.timedelta(days=UPCOMING_DAYS))

    print(f"=== Earnings Momentum Fetch | {today_str} | Lookback {LOOKBACK_YEARS}y ===\n")

    # ── Universe (curated watchlist) ──
    tickers = load_tickers()
    print(f"Universe: {len(tickers)} watchlist tickers\n")

    # ── Fetch earnings ──
    earnings_hist = {}
    upcoming_earnings = []
    today_earnings = []
    earn_dates = {}
    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {sym}...", end=" ", flush=True)
        res = fetch_earnings(sym, today_str)
        raw = res['history']

        for d in res['upcoming']:
            if d <= upcoming_cutoff:
                upcoming_earnings.append({
                    'symbol': sym,
                    'date': d,
                    'daysUntil': (datetime.datetime.strptime(d, '%Y-%m-%d').date() - today).days
                })
        if any(e['date'] == today_str for e in raw) or today_str in res['upcoming']:
            today_earnings.append({'symbol': sym, 'date': today_str})

        # All known earnings dates (history + upcoming) drive the next-earnings exit.
        all_dates = sorted({e['date'] for e in raw} | set(res['upcoming']))
        if all_dates:
            earn_dates[sym] = all_dates

        filtered = [e for e in raw if e['date'] >= cutoff_str]
        if filtered:
            earnings_hist[sym] = filtered
            print(f"{len(filtered)} events")
        else:
            print("no data")
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    upcoming_earnings.sort(key=lambda x: (x['date'], x['symbol']))

    # ── Fetch prices (full universe + benchmark) ──
    price_syms = sorted(set(earnings_hist) | {BENCHMARK})
    print(f"\nFetching prices for {len(price_syms)} symbols (incl. {BENCHMARK})...")
    prices, price_idx = {}, {}
    for i, sym in enumerate(price_syms):
        print(f"  [{i+1}/{len(price_syms)}] {sym}...", end=" ", flush=True)
        p = fetch_prices(sym, cutoff_str)
        if p:
            prices[sym] = p
            price_idx[sym] = _build_index(p)
            print(f"{len(p)} bars")
        else:
            print("no prices")
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    bench_idx = price_idx.get(BENCHMARK)
    if not bench_idx:
        print(f"WARNING: no {BENCHMARK} prices — abnormal returns unavailable")

    # ── Build price-scaled SUE events (long only on positive SUE) ──
    events = []
    skipped_low_sue = 0
    for sym, hist in earnings_hist.items():
        idx = price_idx.get(sym)
        if not idx:
            continue
        for e in hist:
            px = _price_asof(idx, e['date'])
            if not px or px <= 0:
                continue
            # SUE as a "surprise yield": EPS surprise as a % of share price.
            sue = round(e['surprise'] / px * 100, 4)
            # PEAD is directional: only go long when the surprise clears a
            # positive threshold. Skip negative/marginal-SUE events entirely.
            if sue < MIN_SUE:
                skipped_low_sue += 1
                continue
            events.append({
                'symbol': sym,
                'date': e['date'],
                'actual': e['actual'],
                'estimate': e['estimate'],
                'surprise': e['surprise'],
                'sue': sue,
                'quarter': quarter_of(e['date']),
            })
    events.sort(key=lambda x: x['date'])
    print(f"\nEarnings signals (SUE >= {MIN_SUE}): {len(events)} "
          f"({skipped_low_sue} skipped below threshold)")

    if not events:
        preserve_last_known_good(today_str, updated_at, upcoming_earnings)
        return

    # ── Simulate trades (one per earnings event) ──
    print("Simulating trades...")
    trades = []
    today_dt = datetime.datetime.strptime(today_str, '%Y-%m-%d')
    for ev in events:
        sym = ev['symbol']
        px = prices.get(sym, [])
        if not px:
            continue

        # Entry: first trading day after earnings.
        entry = next((p for p in px if p['date'] > ev['date']), None)
        if not entry:
            continue

        entry_dt = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
        target_exit_dt = entry_dt + datetime.timedelta(days=HOLD_DAYS)
        exit_reason = 'hold_period'

        # Exit early if the next earnings event lands inside the hold window.
        nxt = next((d for d in earn_dates.get(sym, []) if d > ev['date']), None)
        if nxt:
            nxt_dt = datetime.datetime.strptime(nxt, '%Y-%m-%d') - datetime.timedelta(days=1)
            if nxt_dt < target_exit_dt:
                target_exit_dt = nxt_dt
                exit_reason = 'next_earnings'
        target_exit_str = target_exit_dt.strftime('%Y-%m-%d')

        bench_entry = _price_asof(bench_idx, entry['date']) if bench_idx else None

        if target_exit_dt > today_dt:
            # Open trade.
            cur = next((p for p in reversed(px) if p['date'] <= today_str), None)
            ret = round(((cur['close'] / entry['close']) - 1) * 100, 2) if cur else None
            path = [round((p['close'] / entry['close'] - 1) * 100, 2)
                    for p in px if entry['date'] <= p['date'] <= today_str]
            bench_ret, abn_ret = _bench_abn(bench_idx, bench_entry,
                                            cur['date'] if cur else None, ret)
            trades.append({
                'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
                'entryDate': entry['date'], 'entryPrice': entry['close'],
                'exitDate': None, 'exitPrice': None,
                'currentPrice': cur['close'] if cur else None,
                'returnPct': ret, 'benchReturnPct': bench_ret, 'abnReturnPct': abn_ret,
                'path': path,
                'daysHeld': (today_dt - entry_dt).days,
                'maxDays': (target_exit_dt - entry_dt).days,
                'exitReason': None, 'open': True
            })
        else:
            # Closed trade.
            exit_bar = next((p for p in reversed(px) if p['date'] <= target_exit_str), None)
            if not exit_bar:
                continue
            ret = round(((exit_bar['close'] / entry['close']) - 1) * 100, 2)
            path = [round((p['close'] / entry['close'] - 1) * 100, 2)
                    for p in px if entry['date'] <= p['date'] <= exit_bar['date']]
            bench_ret, abn_ret = _bench_abn(bench_idx, bench_entry, exit_bar['date'], ret)
            trades.append({
                'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
                'entryDate': entry['date'], 'entryPrice': entry['close'],
                'exitDate': exit_bar['date'], 'exitPrice': exit_bar['close'],
                'currentPrice': None,
                'returnPct': ret, 'benchReturnPct': bench_ret, 'abnReturnPct': abn_ret,
                'path': path,
                'daysHeld': (datetime.datetime.strptime(exit_bar['date'], '%Y-%m-%d') - entry_dt).days,
                'maxDays': HOLD_DAYS, 'exitReason': exit_reason, 'open': False
            })

    open_t = [t for t in trades if t['open']]
    closed_t = [t for t in trades if not t['open']]
    closed_rets = [t['returnPct'] for t in closed_t if t['returnPct'] is not None]
    closed_abn = [t['abnReturnPct'] for t in closed_t if t['abnReturnPct'] is not None]
    closed_bench = [t['benchReturnPct'] for t in closed_t if t['benchReturnPct'] is not None]
    wins = [r for r in closed_rets if r > 0]
    losses = [r for r in closed_rets if r <= 0]

    trade_stats = {
        'total': len(trades),
        'open': len(open_t),
        'closed': len(closed_t),
        'winRate': round(len(wins) / len(closed_rets) * 100, 1) if closed_rets else 0,
        'avgReturn': round(sum(closed_rets) / len(closed_rets), 2) if closed_rets else 0,
        'avgWin': round(sum(wins) / len(wins), 2) if wins else 0,
        'maxWin': round(max(wins), 2) if wins else 0,
        'avgLoss': round(sum(losses) / len(losses), 2) if losses else 0,
        'maxLoss': round(min(losses), 2) if losses else 0,
        # Market-adjusted (abnormal vs benchmark) — the metric that reflects drift.
        'avgAbnReturn': round(sum(closed_abn) / len(closed_abn), 2) if closed_abn else None,
        'avgBenchReturn': round(sum(closed_bench) / len(closed_bench), 2) if closed_bench else None,
        'abnWinRate': round(len([r for r in closed_abn if r > 0]) / len(closed_abn) * 100, 1) if closed_abn else None,
        'tStat': _tstat(closed_abn),
    }

    print(f"Total: {len(trades)} ({len(open_t)} open, {len(closed_t)} closed)")
    if closed_rets:
        print(f"Win rate: {trade_stats['winRate']}%  Avg raw: {trade_stats['avgReturn']}%  "
              f"Avg abnormal: {trade_stats['avgAbnReturn']}%  t={trade_stats['tStat']}")

    # ── Today's activity ──
    today_activity = {
        'date': today_str,
        'earnings': today_earnings,
        'entered': [t for t in trades if t['entryDate'] == today_str],
        'exited': [t for t in trades if t.get('exitDate') == today_str],
    }

    # ── Live data-source status (shown in the dashboard header) ──
    n_priced = len([s for s in prices if s != BENCHMARK])
    data_sources = [
        {'name': 'Yahoo Finance', 'detail': 'earnings & prices',
         'live': bool(prices), 'count': f'{n_priced}/{len(tickers)} tickers'},
        {'name': f'Yahoo Finance · {BENCHMARK}', 'detail': 'benchmark',
         'live': bool(bench_idx), 'count': None},
        {'name': 'Watchlist', 'detail': 'edit on GitHub',
         'live': True, 'count': f'{len(tickers)} tickers', 'url': WATCHLIST_URL},
    ]

    output = {
        'updated': today_str,
        'updatedAt': updated_at,
        'config': {
            'lookbackYears': LOOKBACK_YEARS,
            'holdDays': HOLD_DAYS,
            'universe': 'Custom watchlist',
            'benchmark': BENCHMARK,
            'sue': 'EPS surprise / price at earnings (%)',
            'selection': f'long on SUE >= {MIN_SUE}',
        },
        'tradeStats': trade_stats,
        'dataSources': data_sources,
        'upcomingEarnings': upcoming_earnings,
        'today': today_activity,
        'trades': trades
    }
    with open('data.json', 'w') as f:
        json.dump(output, f)
    print(f"\nWrote data.json ({len(json.dumps(output))} bytes). Done.")


def _bench_abn(bench_idx, bench_entry, exit_date, ret):
    """Benchmark return over [entry, exit] and abnormal return (stock - bench)."""
    if not bench_idx or bench_entry is None or not exit_date or ret is None:
        return None, None
    bench_exit = _price_asof(bench_idx, exit_date)
    if not bench_exit or bench_entry <= 0:
        return None, None
    bench = round((bench_exit / bench_entry - 1) * 100, 2)
    return bench, round(ret - bench, 2)


def preserve_last_known_good(today_str, updated_at, upcoming_earnings):
    """A refresh that returns zero events is almost always a transient upstream
    block (Yahoo rate-limiting the runner), not a real "no trades" state.

    Rather than overwriting good data with an empty error page — which blanks the
    dashboard — keep the last-known-good trades and flag them stale so the UI can
    say so. Only fall back to the empty error payload if there's no prior data."""
    try:
        with open('data.json') as f:
            prev = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        prev = None

    if prev and prev.get('trades') and not prev.get('error'):
        prev['updatedAt'] = updated_at
        prev['stale'] = True
        prev['dataWarning'] = (
            "Live refresh returned no data (Yahoo Finance is likely rate-limiting "
            "this run); showing last-known-good trades from "
            f"{prev.get('updated', 'a previous run')}."
        )
        # Upcoming earnings don't depend on price history, so refresh if we got any.
        if upcoming_earnings:
            prev['upcomingEarnings'] = upcoming_earnings
        with open('data.json', 'w') as f:
            json.dump(prev, f)
        print(f"No new data — preserved {len(prev['trades'])} last-known-good "
              "trades (marked stale).")
    else:
        with open('data.json', 'w') as f:
            json.dump({'trades': [], 'updated': today_str, 'updatedAt': updated_at,
                       'upcomingEarnings': upcoming_earnings, 'error': 'No data'}, f)
        print("No data and no prior snapshot. Wrote empty data.json.")


def _tstat(xs):
    """One-sample t-stat of a return series vs zero (None if too few points)."""
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return None
    return round(mean / math.sqrt(var / n), 2)


if __name__ == '__main__':
    main()
