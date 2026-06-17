#!/usr/bin/env python3
"""
Fetch earnings surprise data + historical prices for a curated watchlist via
yfinance. Simulates a PEAD strategy and outputs per-trade data + summary stats
to data.json.

Methodology notes (see README / PR for rationale):
  * Universe: a hand-curated watchlist in tickers.txt (one symbol per line,
    '#' comments allowed). Edit that file to change which stocks are tracked.
  * Each earnings event becomes a long trade only when its SUE clears a
    positive threshold (MIN_SUE) AND the reported EPS is itself positive;
    negative/marginal-SUE events are skipped, as are "less-bad loss" beats
    (a positive surprise on a still-negative EPS). PEAD is directional, so
    trading non-positive surprises long just fights the drift.
  * SUE is the EPS surprise scaled by the stock's price at the earnings date
    (a unitless "surprise yield"), used both to gate entry and for display.
  * Returns are reported both raw and market-adjusted (abnormal = stock - SPY
    over the same holding window), with a t-stat on the abnormal returns.
"""

import json
import math
import os
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


def _finite(x):
    """True only for a real, finite number (not None, NaN, or +/-Inf)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN, Infinity) with None.

    Python's json emits these as the bare tokens `NaN`/`Infinity`, which are
    valid for Python's own loader but rejected by JavaScript's JSON.parse — a
    single NaN anywhere in data.json blanks the entire dashboard. Sanitizing
    here keeps the output parseable everywhere."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _dump_json(obj, f):
    """Write JSON guaranteed to be browser-parseable (no NaN/Infinity tokens)."""
    json.dump(_json_safe(obj), f, allow_nan=False)


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

# How far back to look for *new* earnings events each run. We don't re-scan all
# of history — only events newer than this (and not already in the log) are
# treated as candidates for a new trade. A quarter-plus of slack means a few
# missed runs (or a paused cron) won't drop a freshly reported quarter.
NEW_EVENT_LOOKBACK_DAYS = 120

# Minimum SUE (EPS surprise as a % of share price) required to enter a long.
# PEAD is directional — positive surprises drift up — so we go long on any
# strictly positive surprise and skip flat/negative ones (i.e. SUE > MIN_SUE).
MIN_SUE = 0.0

BENCHMARK = "SPY"        # broad-market proxy for abnormal-return calculation

# Persistent, ever-growing log of every trade we've ever opened, keyed by
# (symbol, earningsDate). Each run only scans for NEW earnings events and
# refreshes the trades still open; closed trades are immutable and are kept
# forever without being re-fetched or re-simulated. The dashboard renders the
# whole log, so history accumulates continuously instead of being rebuilt from
# a trailing window each run.
TRADE_LOG_FILE = "trades_log.json"

# Curated, user-editable watchlist. The dashboard's "Watchlist" chip links here.
TICKERS_FILE = "tickers.txt"
WATCHLIST_URL = (
    "https://github.com/hansenvalueinvesting/post-earnings-announcement-drift/"
    "blob/main/tickers.txt"
)
# The dashboard's "Trade Log" chip links to the persistent log on GitHub.
TRADE_LOG_URL = (
    "https://github.com/hansenvalueinvesting/post-earnings-announcement-drift/"
    "blob/main/trades_log.json"
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
        # Drop bars Yahoo returns with a missing/NaN close (it occasionally
        # serves partial or placeholder rows). A NaN close otherwise poisons
        # every return derived from it and, once written, produces a literal
        # `NaN` token that JavaScript's JSON.parse rejects.
        return [{'date': d.strftime('%Y-%m-%d'), 'close': round(float(r['Close']), 2)}
                for d, r in df.iterrows() if _finite(r['Close'])]
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
    upcoming_cutoff = str(today + datetime.timedelta(days=UPCOMING_DAYS))

    print(f"=== Earnings Momentum Fetch | {today_str} | incremental (new events only) ===\n")

    # ── Universe (curated watchlist) ──
    tickers = load_tickers()
    print(f"Universe: {len(tickers)} watchlist tickers\n")

    # ── Load the persistent log (every trade we've ever opened) ──
    log = load_trade_log()
    open_trades = [t for t in log.values() if t.get('open')]
    print(f"Trade log: {len(log)} trades on file ({len(open_trades)} still open).\n")

    # ── Scan earnings metadata for the whole watchlist (cheap, no prices) ──
    # Used only to surface upcoming earnings, drive next-earnings exits, and spot
    # NEW events — recent earnings we haven't already logged a trade for.
    new_event_floor = str(today - datetime.timedelta(days=NEW_EVENT_LOOKBACK_DAYS))
    # A ticker freshly added to the watchlist has no trades on file, so the
    # shallow new-event window would only ever catch its most recent quarter.
    # Backfill such tickers across the full LOOKBACK_YEARS the first time we see
    # them, so their history matches the rest of the watchlist; established
    # tickers (already in the log) keep the cheap shallow scan.
    backfill_floor = str(today - datetime.timedelta(days=365 * LOOKBACK_YEARS + 7))
    logged_syms = {s for (s, _) in log}
    # One-off full re-seed (FULL_RESEED env var): scan the full LOOKBACK_YEARS
    # for *every* ticker, not just brand-new ones. Used after changing the
    # selection rule (e.g. lowering MIN_SUE) so previously-skipped events get a
    # chance to enter. Adding is the common case (only events not already in the
    # log become candidates), but a re-seed also *reconciles* existing trades
    # against the current rule: trades opened on a now-disqualified event (a
    # non-positive reported EPS) are pruned, since we re-fetch the full earnings
    # metadata anyway. Normal incremental runs never prune.
    full_reseed = os.environ.get('FULL_RESEED', '').strip().lower() in ('1', 'true', 'yes')
    if full_reseed:
        print("FULL_RESEED set — scanning the full LOOKBACK_YEARS for every "
              "watchlist ticker (one-off backfill + reconcile).\n")
    upcoming_earnings = []
    today_earnings = []
    earn_dates = {}
    new_candidates = []
    # Reported EPS per scanned event, used to prune disqualified logged trades on
    # a re-seed. Only events we actually fetched land here, so a fetch miss can
    # never cause a trade to be pruned.
    eps_actual = {}
    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {sym}...", end=" ", flush=True)
        res = fetch_earnings(sym, today_str)
        raw = res['history']
        for e in raw:
            eps_actual[(sym, e['date'])] = e['actual']

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

        # New events = earnings not already represented by a logged trade. A
        # ticker already in the log only needs the shallow window; a brand-new
        # one (or any ticker during a full re-seed) is scanned across the full
        # backfill window.
        deep = full_reseed or sym not in logged_syms
        floor = backfill_floor if deep else new_event_floor
        fresh = [e for e in raw
                 if e['date'] >= floor and (sym, e['date']) not in log]
        new_candidates.extend({'symbol': sym, **e} for e in fresh)
        print(f"{len(fresh)} new" if fresh else "—")
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    upcoming_earnings.sort(key=lambda x: (x['date'], x['symbol']))

    # ── Fetch prices only for what we must (re)simulate ──
    # That's the trades still open (to refresh / finalize them) plus any new
    # events. Closed trades already in the log are immutable and never re-priced.
    today_dt = datetime.datetime.strptime(today_str, '%Y-%m-%d')
    # Earliest date each symbol must cover: an open trade's entry date, or a new
    # event's earnings date. Tracking it per symbol keeps a deep backfill (a
    # freshly added ticker reaching back LOOKBACK_YEARS) from forcing the same
    # expensive 5-year pull on every other symbol.
    sym_floor = {}
    for sym, ds in [(t['symbol'], t['entryDate']) for t in open_trades] + \
                   [(e['symbol'], e['date']) for e in new_candidates]:
        if sym not in sym_floor or ds < sym_floor[sym]:
            sym_floor[sym] = ds
    need_syms = set(sym_floor)

    prices, price_idx, bench_idx = {}, {}, None
    if need_syms:
        # The benchmark must span the whole window so every trade's abnormal
        # return can be computed, so fetch it from the earliest date we need.
        global_floor = min(sym_floor.values())

        def price_start_for(sym):
            # Start a week before the earliest thing we need so as-of lookups
            # (entry price, price-at-earnings for SUE) land on a real bar.
            d = global_floor if sym == BENCHMARK else sym_floor[sym]
            start_dt = datetime.datetime.strptime(d, '%Y-%m-%d') - datetime.timedelta(days=7)
            return start_dt.strftime('%Y-%m-%d')

        price_syms = sorted(need_syms | {BENCHMARK})
        print(f"\nFetching prices for {len(price_syms)} symbols (incl. {BENCHMARK}) "
              f"from {global_floor} — {len(open_trades)} open + {len(new_candidates)} new...")
        for i, sym in enumerate(price_syms):
            print(f"  [{i+1}/{len(price_syms)}] {sym}...", end=" ", flush=True)
            p = fetch_prices(sym, price_start_for(sym))
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
    else:
        print("\nNo open trades and no new events — log already current.")

    # ── Build the events to (re)simulate ──
    # New events become trades only if their SUE clears the threshold; open
    # trades are re-simulated to refresh their value and finalize them on close.
    sim_events = []
    skipped_low_sue = skipped_neg_eps = 0
    for e in new_candidates:
        idx = price_idx.get(e['symbol'])
        if not idx:
            continue
        # Only trade genuinely profitable quarters. A positive SUE can come from
        # a loss that merely beat an even worse estimate (e.g. -10 vs -20 EPS);
        # that's still a money-losing report, so skip any non-positive reported EPS.
        if e['actual'] <= 0:
            skipped_neg_eps += 1
            continue
        px_at = _price_asof(idx, e['date'])
        if not px_at or px_at <= 0:
            continue
        # SUE as a "surprise yield": EPS surprise as a % of share price.
        sue = round(e['surprise'] / px_at * 100, 4)
        # PEAD is directional: only go long on strictly positive surprises.
        if sue <= MIN_SUE:
            skipped_low_sue += 1
            continue
        sim_events.append({'symbol': e['symbol'], 'date': e['date'], 'sue': sue})
    for t in open_trades:
        sim_events.append({'symbol': t['symbol'], 'date': t['earningsDate'], 'sue': t['sue']})

    # ── Simulate and upsert into the log ──
    new_trades = updated = 0
    for ev in sim_events:
        trade = simulate_trade(ev, prices, earn_dates, bench_idx, today_str, today_dt)
        if trade is None:
            continue  # can't price it this run — leave any existing record intact
        key = (ev['symbol'], ev['date'])
        if key in log:
            updated += 1
        else:
            new_trades += 1
        log[key] = trade

    # On a re-seed, reconcile existing trades against the current rule: drop any
    # opened on an event whose reported EPS we now know to be non-positive. Only
    # events fetched this run (in eps_actual) are eligible, so an unscanned or
    # failed-to-fetch trade is left untouched rather than wrongly pruned.
    pruned = 0
    if full_reseed:
        for key in [k for k in log if eps_actual.get(k, 1) <= 0]:
            del log[key]
            pruned += 1

    trades = save_trade_log(log)
    print(f"\nSimulated {len(sim_events)} events: +{new_trades} new, {updated} refreshed, "
          f"{pruned} pruned (non-positive EPS), "
          f"{skipped_low_sue} with non-positive SUE (<= {MIN_SUE:g}), "
          f"{skipped_neg_eps} with non-positive reported EPS. "
          f"Log now holds {len(trades)} trades.")

    if not trades:
        preserve_last_known_good(today_str, updated_at, upcoming_earnings)
        return

    open_t = [t for t in trades if t['open']]
    closed_t = [t for t in trades if not t['open']]
    closed_rets = [t['returnPct'] for t in closed_t if _finite(t['returnPct'])]
    closed_abn = [t['abnReturnPct'] for t in closed_t if _finite(t['abnReturnPct'])]
    closed_bench = [t['benchReturnPct'] for t in closed_t if _finite(t['benchReturnPct'])]
    wins = [r for r in closed_rets if r > 0]
    losses = [r for r in closed_rets if r <= 0]

    # Annualize the *average* trade by the average holding period. We avoid
    # annualizing each trade then averaging: the (365/daysHeld) exponent is
    # convex, so it inflates winners far more than it floors losers (one +42%
    # trade annualizes to +752%), dragging the mean well above the median and
    # misrepresenting a typical trade.
    hold_days = [t['daysHeld'] for t in closed_t if _finite(t['returnPct']) and t.get('daysHeld')]
    avg_ret = sum(closed_rets) / len(closed_rets) if closed_rets else None
    avg_days = sum(hold_days) / len(hold_days) if hold_days else None
    avg_ann_return = (((1 + avg_ret / 100) ** (365 / avg_days) - 1) * 100
                      if avg_ret is not None and avg_days else None)

    trade_stats = {
        'total': len(trades),
        'open': len(open_t),
        'closed': len(closed_t),
        'winRate': round(len(wins) / len(closed_rets) * 100, 1) if closed_rets else 0,
        'avgReturn': round(sum(closed_rets) / len(closed_rets), 2) if closed_rets else 0,
        'avgAnnReturn': round(avg_ann_return, 2) if avg_ann_return is not None else None,
        'avgHoldDays': round(avg_days, 1) if avg_days else None,
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
    # On an incremental run we only price open trades + new events, so base the
    # "live" flag on the earnings scan (which covers the whole watchlist).
    yahoo_live = bool(earn_dates) or bool(prices)
    data_sources = [
        {'name': 'Yahoo Finance', 'detail': 'earnings & prices',
         'live': yahoo_live, 'count': f'{len(earn_dates)}/{len(tickers)} tickers'},
        {'name': f'Yahoo Finance · {BENCHMARK}', 'detail': 'benchmark',
         'live': bool(bench_idx) or not need_syms, 'count': None},
        {'name': 'Watchlist', 'detail': 'edit on GitHub',
         'live': True, 'count': f'{len(tickers)} tickers', 'url': WATCHLIST_URL},
        {'name': 'Trade Log', 'detail': 'view on GitHub',
         'live': True, 'count': f'{len(trades)} trades', 'url': TRADE_LOG_URL},
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
            'selection': f'long on SUE > {MIN_SUE:g} & reported EPS > 0',
        },
        'tradeStats': trade_stats,
        'dataSources': data_sources,
        'upcomingEarnings': upcoming_earnings,
        'today': today_activity,
        'trades': trades
    }
    # If we needed live prices (open trades / new events) but got none, Yahoo is
    # likely rate-limiting this run — flag the open trades as possibly stale
    # rather than silently showing last run's prices as current.
    if need_syms and not prices:
        output['stale'] = True
        output['dataWarning'] = (
            "Live price refresh returned no data (Yahoo Finance is likely "
            "rate-limiting this run); open-trade values may be stale."
        )
    with open('data.json', 'w') as f:
        _dump_json(output, f)
    print(f"\nWrote data.json ({len(json.dumps(_json_safe(output)))} bytes). Done.")


def simulate_trade(ev, prices, earn_dates, bench_idx, today_str, today_dt):
    """Simulate one PEAD trade for earnings event `ev` (needs 'symbol','date','sue').

    Entry is the first trading day after earnings; the position is held for
    HOLD_DAYS or until the next earnings date, whichever comes first. Returns a
    trade dict (open or closed) or None if it can't be priced yet (no price
    series, no entry bar, or no exit bar)."""
    sym = ev['symbol']
    px = prices.get(sym, [])
    if not px:
        return None

    # Entry: first trading day after earnings.
    entry = next((p for p in px if p['date'] > ev['date']), None)
    if not entry:
        return None

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
        return {
            'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
            'entryDate': entry['date'], 'entryPrice': entry['close'],
            'exitDate': None, 'exitPrice': None,
            'currentPrice': cur['close'] if cur else None,
            'returnPct': ret, 'benchReturnPct': bench_ret, 'abnReturnPct': abn_ret,
            'path': path,
            'daysHeld': (today_dt - entry_dt).days,
            'maxDays': (target_exit_dt - entry_dt).days,
            'exitReason': None, 'open': True,
        }

    # Closed trade.
    exit_bar = next((p for p in reversed(px) if p['date'] <= target_exit_str), None)
    if not exit_bar:
        return None
    ret = round(((exit_bar['close'] / entry['close']) - 1) * 100, 2)
    path = [round((p['close'] / entry['close'] - 1) * 100, 2)
            for p in px if entry['date'] <= p['date'] <= exit_bar['date']]
    bench_ret, abn_ret = _bench_abn(bench_idx, bench_entry, exit_bar['date'], ret)
    return {
        'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
        'entryDate': entry['date'], 'entryPrice': entry['close'],
        'exitDate': exit_bar['date'], 'exitPrice': exit_bar['close'],
        'currentPrice': None,
        'returnPct': ret, 'benchReturnPct': bench_ret, 'abnReturnPct': abn_ret,
        'path': path,
        'daysHeld': (datetime.datetime.strptime(exit_bar['date'], '%Y-%m-%d') - entry_dt).days,
        'maxDays': HOLD_DAYS, 'exitReason': exit_reason, 'open': False,
    }


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
            _dump_json(prev, f)
        print(f"No new data — preserved {len(prev['trades'])} last-known-good "
              "trades (marked stale).")
    else:
        with open('data.json', 'w') as f:
            _dump_json({'trades': [], 'updated': today_str, 'updatedAt': updated_at,
                        'upcomingEarnings': upcoming_earnings, 'error': 'No data'}, f)
        print("No data and no prior snapshot. Wrote empty data.json.")


def load_trade_log():
    """Load the persistent trade log (every trade ever recorded, open or closed)
    as a dict keyed by (symbol, earningsDate).

    Returns {} if the log doesn't exist yet (first run) or is unreadable."""
    try:
        with open(TRADE_LOG_FILE) as f:
            recs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {(t['symbol'], t['earningsDate']): t for t in recs}


def save_trade_log(log):
    """Persist the trade log (sorted by earnings date) and return the sorted
    list of records.

    Written as a JSON array with one trade per line: still valid JSON, but
    readable as a list and diff-friendly (a new trade is a one-line change)."""
    recs = sorted(log.values(), key=lambda t: (t['earningsDate'], t['symbol']))
    with open(TRADE_LOG_FILE, 'w') as f:
        f.write('[\n')
        f.write(',\n'.join('  ' + json.dumps(_json_safe(r), allow_nan=False) for r in recs))
        f.write('\n]\n' if recs else ']\n')
    return recs


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
