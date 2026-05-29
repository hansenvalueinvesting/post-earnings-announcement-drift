#!/usr/bin/env python3
"""
Fetch NASDAQ-100 + S&P 500 earnings surprise data + historical prices via yfinance.
Simulates a PEAD strategy and outputs per-trade data + summary stats to data.json.

Methodology notes (see README / PR for rationale):
  * Point-in-time membership: signals are only generated for a stock if it was
    actually an index member on the earnings date. The current S&P 500 list is
    rewound through Wikipedia's "changes" table so we don't (a) trade names that
    hadn't yet been added or (b) silently drop names that were later removed
    (survivorship bias). NASDAQ-100-only names are treated as members throughout
    (we lack a clean point-in-time source for them).
  * SUE is the EPS surprise scaled by the stock's price at the earnings date
    (a unitless, cross-sectionally comparable "surprise yield").
  * Ranking uses a TRAILING distribution: a signal is kept only if its SUE is in
    the top decile of signals observed in the prior 365 days -- no look-ahead.
  * Returns are reported both raw and market-adjusted (abnormal = stock - SPY
    over the same holding window), with a t-stat on the abnormal returns.
"""

import json
import math
import bisect
import datetime
import time
from collections import deque
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

LOOKBACK_YEARS = 5
HOLD_DAYS = 60
UPCOMING_DAYS = 30

BENCHMARK = "SPY"        # market proxy for abnormal-return calculation
TRAILING_DAYS = 365      # window for the trailing SUE distribution
MIN_TRAILING = 50        # min prior signals before we trust a percentile
TOP_PCTILE = 0.90        # keep signals at/above this percentile of the trailing window

# NDX-100 tickers (reliable fallback, updated periodically)
NDX100 = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMGN","AMZN",
    "ANSS","APP","ARM","ASML","AVGO","AZN","BIIB","BKNG","BKR","CCEP",
    "CDNS","CDW","CEG","CHTR","CMCSA","COIN","COST","CPRT","CRWD","CSCO",
    "CSGP","CTAS","CTSH","DASH","DDOG","DLTR","DXCM","EA","EXC","FANG",
    "FAST","FTNT","GEHC","GFS","GILD","GOOG","GOOGL","HON","IDXX","ILMN",
    "INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR",
    "MCHP","MDB","MDLZ","MELI","META","MNST","MRNA","MRVL","MSFT","MU",
    "NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD",
    "PEP","PYPL","QCOM","REGN","ROP","ROST","SBUX","SMCI","SNPS","TEAM",
    "TMUS","TSLA","TTD","TTWO","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS"
]

# S&P 500 tickers (reliable fallback, updated periodically). Dots are
# normalized to dashes to match yfinance (e.g. BRK.B -> BRK-B).
SP500 = [
    "A","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM",
    "ADP","ADSK","AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM",
    "ALB","ALGN","ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP",
    "AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD","APH","APTV",
    "ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP","AZO","BA",
    "BAC","BALL","BAX","BBY","BDX","BEN","BF-B","BG","BIIB","BK","BKNG",
    "BKR","BLDR","BLK","BMY","BR","BRK-B","BRO","BSX","BX","BXP",
    "C","CAG","CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL",
    "CDNS","CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI",
    "CINF","CL","CLX","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COIN","COO","COP","COR","COST","CPAY","CPB","CPRT","CPT",
    "CRL","CRM","CRWD","CSCO","CSGP","CSX","CTAS","CTRA","CTSH","CTVA",
    "CVS","CVX","CZR","D","DAL","DASH","DAY","DD","DE","DECK",
    "DELL","DG","DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV",
    "DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXCM","EA","EBAY",
    "ECL","ED","EFX","EG","EIX","EL","ELV","EMN","EMR","ENPH",
    "EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN","ETR",
    "EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG","FAST","FCX",
    "FDS","FDX","FE","FFIV","FI","FICO","FIS","FITB","FOX","FOXA",
    "FRT","FSLR","FTNT","FTV","GD","GDDY","GE","GEHC","GEN","GEV",
    "GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN",
    "GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HES","HIG",
    "HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY",
    "HUBB","HUM","HWM","IBM","ICE","IDXX","IEX","IFF","INCY","INTC",
    "INTU","INVH","IP","IPG","IQV","IR","IRM","ISRG","IT","ITW",
    "IVZ","J","JBHT","JBL","JCI","JKHY","JNJ","JNPR","JPM","K",
    "KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX",
    "KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LII","LIN",
    "LKQ","LLY","LMT","LNT","LOW","LRCX","LULU","LUV","LVS","LW",
    "LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO",
    "MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC",
    "MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MS",
    "MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN",
    "NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP",
    "NTRS","NUE","NVDA","NVR","NWS","NWSA","NXPI","O","ODFL","OKE",
    "OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC","PAYX",
    "PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH","PHM",
    "PKG","PLD","PM","PNC","PNR","PNW","PODD","POOL","PPG","PPL",
    "PRU","PSA","PSX","PTC","PWR","PYPL","QCOM","RCL","REG","REGN",
    "RF","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX",
    "RVTY","SBAC","SBUX","SCHW","SHW","SJM","SLB","SMCI","SNA","SNPS",
    "SO","SOLV","SPG","SPGI","SRE","STE","STLD","STT","STX","STZ",
    "SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY","TECH",
    "TEL","TER","TFC","TFX","TGT","TJX","TMO","TMUS","TPR","TRGP",
    "TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT",
    "TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI",
    "USB","V","VICI","VLO","VLTO","VMC","VRSK","VRSN","VRTX","VST",
    "VTR","VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL",
    "WFC","WM","WMB","WMT","WRB","WST","WTW","WY","WYNN","XEL",
    "XOM","XYL","YUM","ZBH","ZBRA","ZTS"
]

# Fallback S&P 500 membership changes (date, added_ticker, removed_ticker) used
# only when the live Wikipedia "changes" table can't be scraped. Covers major
# removals over the last ~5 years so survivorship add-back still works offline.
# Tickers are normalized to yfinance form (dots -> dashes).
SP500_CHANGES_FALLBACK = [
    ("2025-07-09", "", "JNPR"),
    ("2024-10-01", "", "BBWI"),
    ("2024-09-23", "", "BIO"),
    ("2024-04-03", "", "ZION"),
    ("2024-03-18", "", "ROL"),
    ("2024-03-18", "", "DISH"),
    ("2023-10-18", "", "DXC"),
    ("2023-10-02", "", "LNC"),
    ("2023-08-25", "", "AAP"),
    ("2023-06-20", "", "FRC"),    # First Republic
    ("2023-05-04", "", "SIVB"),   # SVB Financial
    ("2023-03-15", "", "SBNY"),   # Signature Bank
    ("2023-03-15", "", "LUMN"),
    ("2023-02-01", "", "VNO"),
    ("2022-12-19", "", "ABMD"),
    ("2022-10-12", "", "TWTR"),   # Twitter (taken private)
    ("2022-06-02", "", "UA"),
    ("2022-04-04", "", "PBCT"),
    ("2022-02-15", "", "XLNX"),   # Xilinx (acquired by AMD)
    ("2022-01-20", "", "CERN"),   # Cerner (acquired by Oracle)
    ("2021-12-20", "", "KSU"),    # Kansas City Southern
    ("2021-10-12", "", "CXO"),
    ("2021-07-21", "", "ALXN"),   # Alexion (acquired by AstraZeneca)
    ("2021-04-20", "", "VAR"),    # Varian (acquired by Siemens)
    ("2021-03-22", "", "FLIR"),
    ("2021-01-21", "", "FLS"),
    ("2020-12-21", "", "TIF"),    # Tiffany (acquired by LVMH)
    ("2020-10-12", "", "NBL"),
    ("2020-04-03", "", "MAC"),
]


def _scrape_tickers(url, match):
    """Scrape ticker symbols from a Wikipedia table. Returns [] on failure."""
    tables = pd.read_html(
        url, match=match, storage_options={"User-Agent": "Mozilla/5.0"}
    )
    if tables:
        df = tables[0]
        for col in df.columns:
            if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                tickers = [str(t).strip().replace('.', '-')
                           for t in df[col].dropna() if isinstance(t, str)]
                if len(tickers) > 50:
                    return tickers
    return []


def get_ndx100():
    """Try Wikipedia first, fall back to hardcoded list. Returns (tickers, live)."""
    try:
        tickers = _scrape_tickers("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker")
        if tickers:
            print(f"Got {len(tickers)} NASDAQ-100 tickers from Wikipedia")
            return tickers, True
    except Exception as e:
        print(f"NASDAQ-100 Wikipedia failed ({e}), using fallback list")
    return NDX100, False


def get_sp500():
    """Try Wikipedia first, fall back to hardcoded list. Returns (tickers, live)."""
    try:
        tickers = _scrape_tickers(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
        if tickers:
            print(f"Got {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers, True
    except Exception as e:
        print(f"S&P 500 Wikipedia failed ({e}), using fallback list")
    return SP500, False


def _parse_date(s):
    """Parse a free-form date string to ISO 'YYYY-MM-DD', or None."""
    try:
        d = pd.to_datetime(str(s), errors='coerce')
        if pd.isna(d):
            return None
        return d.date().isoformat()
    except Exception:
        return None


def _scrape_sp500_changes():
    """Scrape Wikipedia's S&P 500 'selected changes' table.

    Returns a list of (date_iso, added_ticker, removed_ticker) tuples. Empty on
    failure (caller falls back to SP500_CHANGES_FALLBACK)."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print(f"S&P 500 changes scrape failed ({e})")
        return []

    for df in tables:
        flat = [tuple(str(x) for x in c) if isinstance(c, tuple) else (str(c),)
                for c in df.columns]
        text = ' '.join('|'.join(t).lower() for t in flat)
        if not ('added' in text and 'removed' in text and 'date' in text):
            continue

        date_col = added_col = removed_col = None
        for col, t in zip(df.columns, flat):
            j = '|'.join(t).lower()
            if 'date' in j and date_col is None:
                date_col = col
            if 'added' in j and 'ticker' in j and added_col is None:
                added_col = col
            if 'removed' in j and 'ticker' in j and removed_col is None:
                removed_col = col
        if date_col is None or (added_col is None and removed_col is None):
            continue

        def _cell(row, col):
            if col is None:
                return ''
            v = str(row[col]).strip().replace('.', '-')
            return '' if v.lower() in ('nan', '') else v

        changes = []
        for _, row in df.iterrows():
            d = _parse_date(row[date_col])
            if not d:
                continue
            changes.append((d, _cell(row, added_col), _cell(row, removed_col)))
        if changes:
            print(f"Got {len(changes)} S&P 500 membership changes from Wikipedia")
            return changes
    return []


def build_membership(sp_current, cutoff_str):
    """Build a point-in-time S&P 500 membership checker.

    Returns (events, extra_tickers, live):
      * events: ticker -> sorted list of (date_iso, 'added'|'removed')
      * extra_tickers: removed-since-cutoff names not in the current list, which
        should be added to the fetch universe to undo survivorship bias.
      * live: True if the changes table was scraped from Wikipedia (vs fallback).
    """
    changes = _scrape_sp500_changes()
    live = bool(changes)
    if not live:
        changes = SP500_CHANGES_FALLBACK
        print(f"Using fallback list of {len(changes)} S&P 500 changes")

    events, extra = {}, set()
    for d, add, rem in changes:
        if add:
            events.setdefault(add, []).append((d, 'added'))
        if rem:
            events.setdefault(rem, []).append((d, 'removed'))
            if rem not in sp_current and d >= cutoff_str:
                extra.add(rem)
    for t in events:
        events[t].sort()
    return events, extra, live


def make_membership_checker(ndx_current, sp_current, sp_events):
    """Return is_member(ticker, date_iso) for point-in-time index membership.

    NASDAQ-100 members are assumed to be members throughout the window (no clean
    point-in-time source). S&P 500 membership is reconstructed from `sp_events`:
    the state on `date` equals the 'before' state of the earliest change after
    `date`, else the current membership.
    """
    def is_member(ticker, date_iso):
        if ticker in ndx_current:
            return True
        evs = sp_events.get(ticker)
        if not evs:
            return ticker in sp_current
        after = [e for e in evs if e[0] > date_iso]
        if not after:
            return ticker in sp_current
        return after[0][1] == 'removed'
    return is_member


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

    # ── Universe + point-in-time membership ──
    ndx, ndx_live = get_ndx100()
    sp, sp_live = get_sp500()
    ndx_set, sp_set = set(ndx), set(sp)
    sp_events, extra, changes_live = build_membership(sp_set, cutoff_str)
    is_member = make_membership_checker(ndx_set, sp_set, sp_events)

    tickers = sorted(set(ndx) | set(sp) | extra)
    print(f"Universe: {len(tickers)} tickers "
          f"({len(extra)} removed names added back to undo survivorship bias)\n")

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

    spy_idx = price_idx.get(BENCHMARK)
    if not spy_idx:
        print(f"WARNING: no {BENCHMARK} prices — abnormal returns unavailable")

    # ── Build price-scaled SUE events (membership-filtered) ──
    events = []
    for sym, hist in earnings_hist.items():
        idx = price_idx.get(sym)
        if not idx:
            continue
        for e in hist:
            if not is_member(sym, e['date']):
                continue
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
    print(f"\nMembership-filtered, price-scaled signals: {len(events)}")

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

        spy_entry = _price_asof(spy_idx, entry['date']) if spy_idx else None

        if target_exit_dt > today_dt:
            # Open trade.
            cur = next((p for p in reversed(px) if p['date'] <= today_str), None)
            ret = round(((cur['close'] / entry['close']) - 1) * 100, 2) if cur else None
            path = [round((p['close'] / entry['close'] - 1) * 100, 2)
                    for p in px if entry['date'] <= p['date'] <= today_str]
            bench_ret, abn_ret = _bench_abn(spy_idx, spy_entry,
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
            bench_ret, abn_ret = _bench_abn(spy_idx, spy_entry, exit_bar['date'], ret)
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
        # Market-adjusted (abnormal vs SPY) — the metric that actually reflects drift.
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
         'live': bool(spy_idx), 'count': None},
        {'name': 'Wikipedia', 'detail': 'NASDAQ-100 constituents',
         'live': ndx_live, 'count': f'{len(ndx_set)} names'},
        {'name': 'Wikipedia', 'detail': 'S&P 500 constituents',
         'live': sp_live, 'count': f'{len(sp_set)} names'},
        {'name': 'Wikipedia', 'detail': 'S&P 500 membership changes',
         'live': changes_live, 'count': f'+{len(extra)} restored'},
    ]

    output = {
        'updated': today_str,
        'updatedAt': updated_at,
        'config': {
            'lookbackYears': LOOKBACK_YEARS,
            'holdDays': HOLD_DAYS,
            'universe': 'NASDAQ-100 + S&P 500 (point-in-time)',
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


def _bench_abn(spy_idx, spy_entry, exit_date, ret):
    """Benchmark return over [entry, exit] and abnormal return (stock - bench)."""
    if not spy_idx or spy_entry is None or not exit_date or ret is None:
        return None, None
    spy_exit = _price_asof(spy_idx, exit_date)
    if not spy_exit or spy_entry <= 0:
        return None, None
    bench = round((spy_exit / spy_entry - 1) * 100, 2)
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
