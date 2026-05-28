#!/usr/bin/env python3
"""
Fetch NASDAQ-100 earnings surprise data + historical prices via yfinance.
Outputs data.json consumed by the dashboard.
"""

import json
import math
import datetime
import time
import sys

import yfinance as yf
import pandas as pd

LOOKBACK_YEARS = 3
HOLD_DAYS = 60

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


def get_ndx100():
    """Try Wikipedia first, fall back to hardcoded list."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            match="Ticker",
            storage_options={"User-Agent": "Mozilla/5.0"}
        )
        if tables:
            df = tables[0]
            for col in df.columns:
                if 'ticker' in str(col).lower() or 'symbol' in str(col).lower():
                    tickers = [str(t).strip().replace('.', '-') for t in df[col].dropna() if isinstance(t, str)]
                    if len(tickers) > 50:
                        print(f"Got {len(tickers)} tickers from Wikipedia")
                        return tickers
    except Exception as e:
        print(f"Wikipedia failed ({e}), using fallback list")
    return NDX100


def fetch_earnings(sym):
    """Fetch earnings dates with actual/estimate EPS via yfinance."""
    try:
        tk = yf.Ticker(sym)
        df = tk.get_earnings_dates(limit=40)
        if df is None or df.empty:
            return []

        records = []
        for idx, row in df.iterrows():
            est = row.get('EPS Estimate')
            act = row.get('Reported EPS')
            if pd.isna(est) or pd.isna(act):
                continue
            dt = idx
            if hasattr(dt, 'date'):
                dt = dt.date()
            records.append({
                'date': str(dt)[:10],
                'actual': float(act),
                'estimate': float(est),
                'surprise': round(float(act) - float(est), 4)
            })
        # Sort oldest first
        records.sort(key=lambda x: x['date'])
        return records
    except Exception as e:
        return []


def fetch_prices(sym, start):
    """Fetch daily close prices."""
    try:
        df = yf.Ticker(sym).history(start=start, auto_adjust=True)
        if df is None or df.empty:
            return []
        return [{'date': d.strftime('%Y-%m-%d'), 'close': round(float(r['Close']), 2)}
                for d, r in df.iterrows()]
    except:
        return []


def compute_sue(earnings):
    """SUE = (actual - estimate) / std(prior surprises). Needs >=4 prior."""
    results = []
    for i, e in enumerate(earnings):
        prior = [x['surprise'] for x in earnings[:i]]
        if len(prior) < 4:
            continue
        mean = sum(prior) / len(prior)
        std = math.sqrt(sum((x - mean)**2 for x in prior) / (len(prior) - 1))
        if std < 0.001:
            continue
        results.append({**e, 'sue': round(e['surprise'] / std, 4)})
    return results


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()
    today_str = str(today)
    updated_at = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    cutoff = today - datetime.timedelta(days=LOOKBACK_YEARS * 365)
    cutoff_str = str(cutoff)

    print(f"=== Earnings Momentum Fetch | {today_str} | Lookback {LOOKBACK_YEARS}y ===\n")

    tickers = get_ndx100()
    print(f"Universe: {len(tickers)} tickers\n")

    # ── Fetch earnings ──
    all_events = []
    for i, sym in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {sym}...", end=" ", flush=True)
        raw = fetch_earnings(sym)
        filtered = [e for e in raw if e['date'] >= cutoff_str]
        sue = compute_sue(filtered)
        if sue:
            for s in sue:
                s['symbol'] = sym
            all_events.extend(sue)
            print(f"{len(sue)} signals")
        else:
            print(f"{len(filtered)} events (need >4 for SUE)" if filtered else "no data")
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    print(f"\nTotal SUE events: {len(all_events)}")

    if not all_events:
        with open('data.json', 'w') as f:
            json.dump({'trades': [], 'updated': today_str, 'updatedAt': updated_at, 'error': 'No data'}, f)
        print("No data. Wrote empty data.json.")
        return

    # ── Top decile per quarter ──
    by_q = {}
    for ev in all_events:
        d = datetime.datetime.strptime(ev['date'], '%Y-%m-%d')
        q = f"{d.year}-Q{(d.month-1)//3+1}"
        by_q.setdefault(q, []).append(ev)

    top = []
    for q, evs in by_q.items():
        evs.sort(key=lambda x: x['sue'], reverse=True)
        n = max(1, int(len(evs) * 0.1))
        for e in evs[:n]:
            e['quarter'] = q
            top.append(e)
    top.sort(key=lambda x: x['date'])
    print(f"Top decile signals: {len(top)}\n")

    # ── Fetch prices ──
    syms_needed = list(set(e['symbol'] for e in top))
    print(f"Fetching prices for {len(syms_needed)} symbols...")
    prices = {}
    for i, sym in enumerate(syms_needed):
        print(f"  [{i+1}/{len(syms_needed)}] {sym}...", end=" ", flush=True)
        p = fetch_prices(sym, cutoff_str)
        if p:
            prices[sym] = p
            print(f"{len(p)} bars")
        else:
            print("no prices")
        if (i+1) % 10 == 0:
            time.sleep(0.5)

    # ── Build earnings date index per symbol ──
    earn_dates = {}
    for ev in all_events:
        earn_dates.setdefault(ev['symbol'], set()).add(ev['date'])
    earn_dates = {s: sorted(d) for s, d in earn_dates.items()}

    # ── Simulate trades ──
    print(f"\nSimulating trades...")
    trades = []
    for ev in top:
        sym = ev['symbol']
        px = prices.get(sym, [])
        if not px:
            continue

        # Entry: first trading day after earnings
        entry = next((p for p in px if p['date'] > ev['date']), None)
        if not entry:
            continue

        entry_dt = datetime.datetime.strptime(entry['date'], '%Y-%m-%d')
        target_exit_dt = entry_dt + datetime.timedelta(days=HOLD_DAYS)
        exit_reason = 'hold_period'

        # Check for next earnings
        nxt = next((d for d in earn_dates.get(sym, []) if d > ev['date']), None)
        if nxt:
            nxt_dt = datetime.datetime.strptime(nxt, '%Y-%m-%d') - datetime.timedelta(days=1)
            if nxt_dt < target_exit_dt:
                target_exit_dt = nxt_dt
                exit_reason = 'next_earnings'

        target_exit_str = target_exit_dt.strftime('%Y-%m-%d')
        today_dt = datetime.datetime.strptime(today_str, '%Y-%m-%d')

        if target_exit_dt > today_dt:
            # Open
            cur = next((p for p in reversed(px) if p['date'] <= today_str), None)
            ret = round(((cur['close'] / entry['close']) - 1) * 100, 2) if cur else None
            trades.append({
                'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
                'entryDate': entry['date'], 'entryPrice': entry['close'],
                'exitDate': None, 'exitPrice': None,
                'currentPrice': cur['close'] if cur else None,
                'returnPct': ret,
                'daysHeld': (today_dt - entry_dt).days,
                'maxDays': (target_exit_dt - entry_dt).days,
                'exitReason': None, 'open': True
            })
        else:
            # Closed
            exit_bar = next((p for p in reversed(px) if p['date'] <= target_exit_str), None)
            if not exit_bar:
                continue
            ret = round(((exit_bar['close'] / entry['close']) - 1) * 100, 2)
            trades.append({
                'symbol': sym, 'sue': ev['sue'], 'earningsDate': ev['date'],
                'entryDate': entry['date'], 'entryPrice': entry['close'],
                'exitDate': exit_bar['date'], 'exitPrice': exit_bar['close'],
                'currentPrice': None, 'returnPct': ret,
                'daysHeld': (datetime.datetime.strptime(exit_bar['date'], '%Y-%m-%d') - entry_dt).days,
                'maxDays': HOLD_DAYS, 'exitReason': exit_reason, 'open': False
            })

    open_t = [t for t in trades if t['open']]
    closed_t = [t for t in trades if not t['open']]
    rets = [t['returnPct'] for t in closed_t if t['returnPct'] is not None]

    print(f"Total: {len(trades)} ({len(open_t)} open, {len(closed_t)} closed)")
    if rets:
        print(f"Win rate: {sum(1 for r in rets if r > 0)/len(rets)*100:.1f}%")
        print(f"Avg return: {sum(rets)/len(rets):.2f}%")

    output = {
        'updated': today_str,
        'updatedAt': updated_at,
        'config': {'lookbackYears': LOOKBACK_YEARS, 'holdDays': HOLD_DAYS, 'universe': 'NASDAQ-100'},
        'trades': trades
    }
    with open('data.json', 'w') as f:
        json.dump(output, f)
    print(f"\nWrote data.json ({len(json.dumps(output))} bytes). Done.")


if __name__ == '__main__':
    main()
