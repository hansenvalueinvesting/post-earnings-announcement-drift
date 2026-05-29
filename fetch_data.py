#!/usr/bin/env python3
"""
Fetch Russell 2000 earnings surprise data + historical prices via yfinance.
Simulates a PEAD strategy and outputs per-trade data + summary stats to data.json.

Methodology notes (see README / PR for rationale):
  * Universe: the Russell 2000 small-cap index. Constituents are pulled live from
    the iShares IWM ETF holdings file. Every successful pull refreshes a curated
    fallback list (russell2000_fallback.json) so an offline run can still produce
    data; the dashboard shows when that fallback was last refreshed.
  * Membership is treated as "current members throughout": today's Russell 2000
    list is assumed to be members across the whole lookback window. The index
    only reconstitutes annually (June) and has no clean point-in-time / "changes"
    feed, so unlike a survivorship-corrected approach this carries some
    survivorship bias. (This mirrors how NASDAQ-100 names were handled before.)
  * SUE is the EPS surprise scaled by the stock's price at the earnings date
    (a unitless, cross-sectionally comparable "surprise yield").
  * Ranking uses a TRAILING distribution: a signal is kept only if its SUE is in
    the top decile of signals observed in the prior 365 days -- no look-ahead.
  * Returns are reported both raw and market-adjusted (abnormal = stock - IWM
    over the same holding window), with a t-stat on the abnormal returns.
"""

import csv
import json
import math
import bisect
import datetime
import time
import urllib.request
from collections import deque
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

LOOKBACK_YEARS = 5
HOLD_DAYS = 60
UPCOMING_DAYS = 30

BENCHMARK = "IWM"        # Russell 2000 ETF — market proxy for abnormal returns
TRAILING_DAYS = 365      # window for the trailing SUE distribution
MIN_TRAILING = 50        # min prior signals before we trust a percentile
TOP_PCTILE = 0.90        # keep signals at/above this percentile of the trailing window

# iShares Russell 2000 ETF (IWM) holdings CSV — the live constituent source.
IWM_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)
# Self-refreshing curated fallback. Rewritten on every successful live pull and
# committed alongside data.json, so offline runs use the most recent known list.
FALLBACK_FILE = "russell2000_fallback.json"

# Seed fallback — a curated set of liquid Russell 2000 names used only if the
# live pull fails AND the fallback file is missing/unreadable. In normal
# operation FALLBACK_FILE supersedes this with the full live membership.
RUSSELL2000_SEED = [
    "AAON","ABCB","ABM","ACAD","ACIW","ACLS","ADMA","AEIS","AGYS","AKR",
    "ALEX","ALG","ALKS","AMBA","AMN","AMWD","ANDE","AORT","APAM","APLE",
    "APOG","ARCB","AROC","ASGN","ASTH","ATEN","ATGE","AVAV","AWR","AXNX",
    "BANF","BANR","BCPC","BDC","BGS","BHE","BJRI","BKE","BL","BLMN",
    "BMI","BOOT","BOX","BRC","BRKL","CAKE","CALM","CARG","CARS","CASH",
    "CATY","CBRL","CBU","CCOI","CENT","CENX","CERT","CHCO","CHEF","CLW",
    "CNK","CNMD","COLB","COLL","COOP","CORT","CPK","CRC","CRVL","CSGS",
    "CTRE","CTS","CVCO","CWST","CWT","DAN","DCOM","DDD","DEA","DFIN",
    "DIOD","DNOW","DOCN","DORM","DXPE","ECPG","EGBN","EIG","ELF","ENS",
    "ENV","EPAC","EPRT","ESE","EXLS","EXPO","EXTR","EZPW","FBK","FCF",
    "FCPT","FELE","FFBC","FHB","FIZZ","FORM","FOXF","FSS","FTDR","FUL",
    "FULT","GBX","GEO","GFF","GMS","GNW","GOLF","GPI","GSHD","GTY",
    "HASI","HCC","HCSG","HELE","HI","HLIO","HMN","HNI","HOMB","HUBG",
    "HWKN","ICUI","IIPR","INDB","INSW","IOSP","IPAR","IRDM","ITGR","ITRI",
    "JACK","JBT","JJSF","JOE","KAI","KALU","KFY","KMT","KN","KWR",
    "LAD","LBRT","LCII","LGND","LKFN","LMAT","LNN","LOB","LRN","LXP",
    "MATX","MCY","MGEE","MGY","MMS","MMSI","MODG","MTX","MYRG","NARI",
    "NBHC","NEO","NEOG","NHC","NMIH","NPO","NSIT","NWBI","NWE","OFG",
    "OII","OMCL","ONB","OSIS","OTTR","OXM","PATK","PBH","PECO","PFBC",
    "PFS","PINC","PIPR","PLXS","PNTG","POWL","PRA","PRGS","PRIM","PRK",
    "PTGX","PZZA","QLYS","RAMP","RDN","RDNT","REZI","RMBS","ROCK","RUSHA",
    "RXO","SABR","SAFT","SAH","SANM","SBCF","SCL","SDGR","SFBS","SFNC",
    "SHAK","SHOO","SIGI","SITM","SKT","SKYW","SLAB","SLG","SM","SMPL",
    "SNEX","SONO","SPSC","SPT","SPTN","SR","SSB","SSD","STBA","STC",
    "STEP","STRA","SXI","SXT","TBBK","TCBK","TDS","TFIN","TGNA","THRM",
    "TMP","TNC","TPH","TRMK","TRN","TRNO","TRUP","TTMI","UCBI","UE",
    "UFPI","UMBF","UNF","UPBD","URBN","USLM","USPH","VC","VCEL","VCYT",
    "VECO","VIAV","VICR","VIRT","VRRM","VRTS","VSCO","WABC","WAFD","WD",
    "WERN","WEN","WGO","WHD","WK","WS","WSFS","WT","WWW","XHR",
    "XNCR","XPEL","YELP","ZD","ZWS"
]


def _clean_ticker(t):
    """Normalize a raw ticker to yfinance form, or '' if it isn't an equity symbol."""
    t = str(t).strip().strip('"').upper().replace('.', '-')
    if not t or t in ('-', 'CASH', 'USD'):
        return ''
    if all(c.isalnum() or c == '-' for c in t) and len(t) <= 6 and any(c.isalpha() for c in t):
        return t
    return ''


def _scrape_russell2000():
    """Download current IWM (Russell 2000 ETF) equity holdings from iShares.

    The CSV has a few preamble lines, then a header row beginning with "Ticker",
    then one row per holding, then a footer of disclaimer text. Returns a list of
    yfinance-normalized tickers, or [] on failure."""
    req = urllib.request.Request(IWM_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8-sig", "replace")
    lines = raw.splitlines()

    start = next((i for i, ln in enumerate(lines)
                  if ln.lower().lstrip('"').startswith('ticker')), None)
    if start is None:
        return []

    tickers = []
    for row in csv.DictReader(lines[start:]):
        asset = (row.get('Asset Class') or '').strip().strip('"').lower()
        if asset and asset != 'equity':
            continue
        tk = _clean_ticker(row.get('Ticker') or '')
        if tk:
            tickers.append(tk)
    return sorted(set(tickers))


def _save_fallback(tickers, date_str):
    """Persist the latest live membership as the curated fallback."""
    try:
        with open(FALLBACK_FILE, 'w') as f:
            json.dump({'updated': date_str, 'tickers': sorted(set(tickers))}, f, indent=1)
        print(f"Refreshed {FALLBACK_FILE} ({len(tickers)} names, as of {date_str})")
    except Exception as e:
        print(f"Could not write {FALLBACK_FILE} ({e})")


def _load_fallback():
    """Load the curated fallback. Returns (tickers, updated_date_or_None)."""
    try:
        with open(FALLBACK_FILE) as f:
            d = json.load(f)
        tickers = [t for t in (_clean_ticker(x) for x in d.get('tickers', [])) if t]
        if tickers:
            return sorted(set(tickers)), d.get('updated')
    except Exception as e:
        print(f"Could not read {FALLBACK_FILE} ({e})")
    return sorted(set(RUSSELL2000_SEED)), None


def get_russell2000(today_str):
    """Resolve the Russell 2000 universe.

    Tries the live iShares IWM holdings file first; on success it refreshes the
    curated fallback and returns (tickers, live=True, fallback_date=None). On
    failure it loads the fallback list and returns (tickers, False, its date)."""
    try:
        tickers = _scrape_russell2000()
        if len(tickers) > 1000:
            print(f"Got {len(tickers)} Russell 2000 holdings from iShares IWM")
            _save_fallback(tickers, today_str)
            return tickers, True, None
        print(f"iShares IWM returned only {len(tickers)} names — using fallback")
    except Exception as e:
        print(f"iShares IWM download failed ({e}) — using fallback")
    tickers, date = _load_fallback()
    print(f"Using fallback list of {len(tickers)} names"
          + (f" (last refreshed {date})" if date else " (seed)"))
    return tickers, False, date


def fetch_earnings(sym, today_str):
    """Fetch earnings dates. Returns confirmed history (with EPS) and upcoming dates."""
    try:
        tk = yf.Ticker(sym)
        df = tk.get_earnings_dates(limit=28)
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
        df = yf.Ticker(sym).history(start=start, auto_adjust=True)
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


def select_trailing_decile(events):
    """Keep events whose price-scaled SUE is in the top decile of the TRAILING
    365-day distribution of signals (no look-ahead). Events are expected sorted
    by date ascending. Returns the kept events with a 'quarter' tag."""
    window = deque()      # (date_iso, sue) within the trailing window
    sorted_sue = []       # sue values in the window, kept sorted for percentiles
    kept = []

    for ev in events:
        d = datetime.datetime.strptime(ev['date'], '%Y-%m-%d').date()
        w_start = (d - datetime.timedelta(days=TRAILING_DAYS)).isoformat()

        # Evict signals that have aged out of the trailing window.
        while window and window[0][0] < w_start:
            _, s0 = window.popleft()
            j = bisect.bisect_left(sorted_sue, s0)
            sorted_sue.pop(j)

        # Rank the current event against prior signals only.
        if len(sorted_sue) >= MIN_TRAILING and ev['sue'] > 0:
            rank = bisect.bisect_left(sorted_sue, ev['sue'])
            pctile = rank / len(sorted_sue)
            if pctile >= TOP_PCTILE:
                ev['quarter'] = quarter_of(ev['date'])
                kept.append(ev)

        # Add current signal to the window for future events to rank against.
        bisect.insort(sorted_sue, ev['sue'])
        window.append((ev['date'], ev['sue']))

    return kept


def main():
    now = datetime.datetime.now(ZoneInfo('America/New_York'))
    today = now.date()
    today_str = str(today)
    updated_at = now.strftime('%Y-%m-%d %H:%M:%S ET')
    cutoff = today - datetime.timedelta(days=LOOKBACK_YEARS * 365)
    cutoff_str = str(cutoff)
    upcoming_cutoff = str(today + datetime.timedelta(days=UPCOMING_DAYS))

    print(f"=== Earnings Momentum Fetch | {today_str} | Lookback {LOOKBACK_YEARS}y ===\n")

    # ── Universe (Russell 2000; current members throughout) ──
    russell, russell_live, fallback_date = get_russell2000(today_str)
    tickers = sorted(set(russell))
    print(f"Universe: {len(tickers)} Russell 2000 tickers\n")

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
    # Prices are needed up front because SUE is scaled by the price at the
    # earnings date, so we must price every candidate, not just the selected.
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

    # ── Build price-scaled SUE events ──
    # Every Russell 2000 ticker is treated as a member throughout the window, so
    # no point-in-time membership filter is applied here.
    events = []
    for sym, hist in earnings_hist.items():
        idx = price_idx.get(sym)
        if not idx:
            continue
        for e in hist:
            px = _price_asof(idx, e['date'])
            if not px or px <= 0:
                continue
            events.append({
                'symbol': sym,
                'date': e['date'],
                'actual': e['actual'],
                'estimate': e['estimate'],
                'surprise': e['surprise'],
                # SUE as a "surprise yield": EPS surprise as a % of share price.
                'sue': round(e['surprise'] / px * 100, 4),
            })
    events.sort(key=lambda x: x['date'])
    print(f"\nPrice-scaled signals: {len(events)}")

    if not events:
        with open('data.json', 'w') as f:
            json.dump({'trades': [], 'updated': today_str, 'updatedAt': updated_at,
                       'upcomingEarnings': upcoming_earnings, 'error': 'No data'}, f)
        print("No data. Wrote empty data.json.")
        return

    # ── Trailing top-decile selection (no look-ahead) ──
    top = select_trailing_decile(events)
    top.sort(key=lambda x: x['date'])
    print(f"Top-decile signals (trailing {TRAILING_DAYS}d): {len(top)}\n")

    # ── Simulate trades ──
    print("Simulating trades...")
    trades = []
    today_dt = datetime.datetime.strptime(today_str, '%Y-%m-%d')
    for ev in top:
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
        # Market-adjusted (abnormal vs IWM) — the metric that actually reflects drift.
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
    if russell_live:
        russell_detail = 'Russell 2000 constituents (IWM)'
        russell_count = f'{len(tickers)} names'
    else:
        russell_detail = ('Russell 2000 fallback · refreshed '
                          + (fallback_date or 'unknown'))
        russell_count = f'{len(tickers)} names'
    data_sources = [
        {'name': 'Yahoo Finance', 'detail': 'earnings & prices',
         'live': bool(prices), 'count': f'{n_priced}/{len(tickers)} tickers'},
        {'name': f'Yahoo Finance · {BENCHMARK}', 'detail': 'benchmark',
         'live': bool(bench_idx), 'count': None},
        {'name': 'iShares IWM', 'detail': russell_detail,
         'live': russell_live, 'count': russell_count},
    ]

    output = {
        'updated': today_str,
        'updatedAt': updated_at,
        'config': {
            'lookbackYears': LOOKBACK_YEARS,
            'holdDays': HOLD_DAYS,
            'universe': 'Russell 2000 (current members)',
            'benchmark': BENCHMARK,
            'sue': 'EPS surprise / price at earnings (%)',
            'ranking': f'top decile of trailing {TRAILING_DAYS}d distribution',
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
