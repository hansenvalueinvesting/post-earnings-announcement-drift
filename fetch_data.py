#!/usr/bin/env python3
"""
Fetch NASDAQ-100 earnings surprise data + historical prices via yfinance.
Simulates a PEAD strategy, computes portfolio performance statistics against a
buy & hold QQQ benchmark, and outputs data.json consumed by the dashboard.
"""

import json
import math
import datetime
import time
import statistics

import yfinance as yf
import pandas as pd

LOOKBACK_YEARS = 3
HOLD_DAYS = 60
START_CAPITAL = 100000
UPCOMING_DAYS = 30
TRADING_DAYS = 252
BENCHMARK = "QQQ"

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


def fetch_earnings(sym, today_str):
    """Fetch earnings dates. Returns confirmed history (with EPS) and upcoming dates."""
    try:
        tk = yf.Ticker(sym)
        df = tk.get_earnings_dates(limit=40)
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


def build_daily_series(trades, prices, qqq, today_str, start_capital):
    """Build a daily portfolio equity curve (equal-weight across open positions)
    plus a buy & hold benchmark curve from the QQQ price series."""
    if not qqq:
        return None
    qmap = {p['date']: p['close'] for p in qqq}
    calendar = sorted(d for d in qmap if d <= today_str)
    entries = [t['entryDate'] for t in trades if t.get('entryDate')]
    if not entries or len(calendar) < 2:
        return None
    start_date = min(entries)
    calendar = [d for d in calendar if d >= start_date]
    if len(calendar) < 2:
        return None

    pmap = {sym: {p['date']: p['close'] for p in plist} for sym, plist in prices.items()}
    windows = []
    for t in trades:
        sym = t['symbol']
        if sym not in pmap:
            continue
        ed = t['entryDate']
        xd = t['exitDate'] if not t['open'] else today_str
        if ed and xd:
            windows.append((sym, ed, xd))

    q0 = qmap[calendar[0]]
    eq_dates = [calendar[0]]
    eq_port = [start_capital]
    eq_bench = [start_capital]
    port_rets, bench_rets = [], []

    for i in range(1, len(calendar)):
        prev, cur = calendar[i - 1], calendar[i]
        day_rets = []
        for sym, ed, xd in windows:
            if ed <= prev and cur <= xd:
                pm = pmap[sym]
                if prev in pm and cur in pm and pm[prev] > 0:
                    day_rets.append(pm[cur] / pm[prev] - 1)
        pr = sum(day_rets) / len(day_rets) if day_rets else 0.0
        br = qmap[cur] / qmap[prev] - 1
        port_rets.append(pr)
        bench_rets.append(br)
        eq_dates.append(cur)
        eq_port.append(round(eq_port[-1] * (1 + pr), 2))
        eq_bench.append(round(start_capital * qmap[cur] / q0, 2))

    return {
        'dates': eq_dates, 'portfolio': eq_port, 'benchmark': eq_bench,
        'port_rets': port_rets, 'bench_rets': bench_rets,
        'start_date': calendar[0], 'end_date': calendar[-1]
    }


def compute_stats(equity, rets, bench_rets, start_date, end_date, start_capital):
    """Compute portfolio performance metrics from a daily equity curve."""
    n = len(rets)
    final = equity[-1]
    total_ret = final / start_capital - 1
    days = (datetime.datetime.strptime(end_date, '%Y-%m-%d')
            - datetime.datetime.strptime(start_date, '%Y-%m-%d')).days
    years = days / 365.25 if days > 0 else 0
    cagr = (final / start_capital) ** (1 / years) - 1 if years > 0 and final > 0 else 0

    mean_d = sum(rets) / n if n else 0
    std_d = statistics.pstdev(rets) if n > 1 else 0
    sharpe = (mean_d / std_d) * math.sqrt(TRADING_DAYS) if std_d > 0 else 0
    downside = math.sqrt(sum(min(r, 0) ** 2 for r in rets) / n) if n else 0
    sortino = (mean_d / downside) * math.sqrt(TRADING_DAYS) if downside > 0 else 0

    peak, mdd = equity[0], 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = v / peak - 1 if peak > 0 else 0
        if dd < mdd:
            mdd = dd
    calmar = cagr / abs(mdd) if mdd < 0 else 0

    beta = 0.0
    if bench_rets and len(bench_rets) == n and n > 1:
        mb = sum(bench_rets) / n
        var_b = sum((b - mb) ** 2 for b in bench_rets) / n
        cov = sum((rets[i] - mean_d) * (bench_rets[i] - mb) for i in range(n)) / n
        beta = cov / var_b if var_b > 0 else 0

    cvar = 0.0
    if n:
        srt = sorted(rets)
        k = max(1, int(math.ceil(n * 0.05)))
        cvar = sum(srt[:k]) / k

    return {
        'finalValue': round(final, 2),
        'totalReturn': round(total_ret * 100, 2),
        'cagr': round(cagr * 100, 2),
        'beta': round(beta, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'maxDrawdown': round(mdd * 100, 2),
        'cvar95': round(cvar * 100, 2),
        'years': round(years, 1)
    }


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()
    today_str = str(today)
    updated_at = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    cutoff = today - datetime.timedelta(days=LOOKBACK_YEARS * 365)
    cutoff_str = str(cutoff)
    upcoming_cutoff = str(today + datetime.timedelta(days=UPCOMING_DAYS))

    print(f"=== Earnings Momentum Fetch | {today_str} | Lookback {LOOKBACK_YEARS}y ===\n")

    tickers = get_ndx100()
    print(f"Universe: {len(tickers)} tickers\n")

    # ── Fetch earnings ──
    all_events = []
    upcoming_earnings = []
    today_earnings = []
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

    upcoming_earnings.sort(key=lambda x: (x['date'], x['symbol']))
    print(f"\nTotal SUE events: {len(all_events)} | Upcoming earnings: {len(upcoming_earnings)}")

    if not all_events:
        with open('data.json', 'w') as f:
            json.dump({'trades': [], 'updated': today_str, 'updatedAt': updated_at,
                       'upcomingEarnings': upcoming_earnings, 'error': 'No data'}, f)
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

    print(f"\nFetching {BENCHMARK} benchmark...")
    qqq = fetch_prices(BENCHMARK, cutoff_str)
    print(f"{len(qqq)} bars")

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

    # ── Today's activity ──
    today_activity = {
        'date': today_str,
        'earnings': today_earnings,
        'entered': [t for t in trades if t['entryDate'] == today_str],
        'exited': [t for t in trades if t.get('exitDate') == today_str],
    }

    # ── Portfolio + benchmark stats ──
    stats = None
    benchmark = None
    equity_out = None
    series = build_daily_series(trades, prices, qqq, today_str, START_CAPITAL)
    if series:
        wins = sum(1 for r in rets if r > 0)
        stats = compute_stats(series['portfolio'], series['port_rets'],
                              series['bench_rets'], series['start_date'],
                              series['end_date'], START_CAPITAL)
        stats['trades'] = len(closed_t)
        stats['winRate'] = round(wins / len(rets) * 100, 1) if rets else 0
        benchmark = compute_stats(series['benchmark'], series['bench_rets'],
                                 series['bench_rets'], series['start_date'],
                                 series['end_date'], START_CAPITAL)
        benchmark['beta'] = 1.0
        equity_out = {'dates': series['dates'],
                      'portfolio': series['portfolio'],
                      'benchmark': series['benchmark']}
        print(f"\nFinal value: ${stats['finalValue']:,.0f} "
              f"(B&H ${benchmark['finalValue']:,.0f}) | CAGR {stats['cagr']}% | "
              f"Sharpe {stats['sharpe']} | Beta {stats['beta']}")

    output = {
        'updated': today_str,
        'updatedAt': updated_at,
        'config': {'lookbackYears': LOOKBACK_YEARS, 'holdDays': HOLD_DAYS,
                   'universe': 'NASDAQ-100', 'startCapital': START_CAPITAL,
                   'benchmark': BENCHMARK},
        'stats': stats,
        'benchmark': benchmark,
        'equity': equity_out,
        'upcomingEarnings': upcoming_earnings,
        'today': today_activity,
        'trades': trades
    }
    with open('data.json', 'w') as f:
        json.dump(output, f)
    print(f"\nWrote data.json ({len(json.dumps(output))} bytes). Done.")


if __name__ == '__main__':
    main()
