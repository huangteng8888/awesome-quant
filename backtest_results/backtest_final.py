import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TERM'] = 'dumb'
from dotenv import load_dotenv; load_dotenv('/home/ht/github/TradingAgents/.env')
import pandas as pd
import numpy as np
import pickle

with open('/tmp/stock_history.pkl', 'rb') as f:
    all_data = pickle.load(f)

with open('/tmp/backtest_results.pkl', 'rb') as f:
    prev_results = pickle.load(f)

def calc_indicators(df):
    close = df['close']; high = df['high']; low = df['low']
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()
    df['sma200'] = close.rolling(200).mean()
    df['ema10'] = close.ewm(span=10, adjust=False).mean()
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['rsi'] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['boll_mid'] = close.rolling(20).mean()
    boll_std = close.rolling(20).std()
    df['boll_ub'] = df['boll_mid'] + 2.0 * boll_std
    df['boll_lb'] = df['boll_mid'] - 2.0 * boll_std
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df

def backtest(df, signals, initial_capital=100000.0):
    df = df.copy(); df['signal'] = signals
    cash = float(initial_capital); shares = 0; trades = []; equity_curve = [float(initial_capital)]
    for i in range(len(df)):
        sig = df['signal'].iloc[i]; price = float(df['close'].iloc[i])
        if sig == 1 and cash > 0:
            buy_shares = int((cash * 0.95) / price)
            if buy_shares > 0: cash -= float(buy_shares)*price; shares += buy_shares; trades.append({'type':'BUY','price':price})
        elif sig == -1 and shares > 0:
            sell_shares = int(shares * 0.95)
            if sell_shares > 0: cash += float(sell_shares)*price; shares -= sell_shares; trades.append({'type':'SELL','price':price})
        equity_curve.append(cash + float(shares) * price)
    final_equity = cash + float(shares) * float(df['close'].iloc[-1])
    equity_curve = np.array(equity_curve, dtype=float)
    rets = np.diff(equity_curve) / equity_curve[:-1]; rets = rets[np.isfinite(rets)]
    total_return = (final_equity - initial_capital) / initial_capital * 100.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    max_dd = 0.0; peak = float(equity_curve[0])
    for eq in equity_curve:
        if float(eq) > peak: peak = float(eq)
        dd = (peak - float(eq)) / peak
        if dd > max_dd: max_dd = dd
    closed = [t for t in trades if t['type']=='SELL']
    win_rate = 0.0
    if len(closed) >= 1:
        buy_p = float(initial_capital); wins = []; losses = []
        for t in closed:
            ret = (float(t['price']) - buy_p) / buy_p * 100.0
            if ret > 0: wins.append(ret)
            else: losses.append(ret)
            buy_p = float(t['price'])
        denom = max(len(wins)+len(losses),1)
        win_rate = float(len(wins)) / denom * 100.0
    return {'total_return':float(total_return),'sharpe':float(sharpe),'max_drawdown':float(max_dd)*100.0,'num_trades':len(closed),'win_rate':float(win_rate)}

def strat_majority_vote(df):
    close = df['close']
    sig_rsi = pd.Series(0, index=df.index)
    sig_rsi.loc[df['rsi'] < 30] = 1; sig_rsi.loc[df['rsi'] > 70] = -1
    sig_macd = pd.Series(0, index=df.index)
    sig_macd.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())] = 1
    sig_macd.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())] = -1
    sig_boll = pd.Series(0, index=df.index)
    sig_boll.loc[df['close'] < df['boll_lb']] = 1; sig_boll.loc[df['close'] > df['boll_ub']] = -1
    sig_ma = pd.Series(0, index=df.index)
    sig_ma.loc[(close > df['sma50']) & (close.shift() <= df['sma50'].shift())] = 1
    sig_ma.loc[(close < df['sma50']) & (close.shift() >= df['sma50'].shift())] = -1
    sig_trend = pd.Series(0, index=df.index)
    uptrend = (close > df['sma50']) & (df['rsi'] > 50) & (df['macd'] > 0)
    sig_trend.loc[uptrend] = 1; sig_trend.loc[~uptrend] = -1
    votes = sig_rsi + sig_macd + sig_boll + sig_ma + sig_trend
    combined = pd.Series(0, index=df.index)
    combined.loc[votes >= 2] = 1; combined.loc[votes <= -2] = -1
    return combined

def strat_consensus(df):
    sig = pd.Series(0, index=df.index)
    macd_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
    macd_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
    sig.loc[(df['rsi'] < 35) & macd_up] = 1
    sig.loc[(df['rsi'] > 65) | macd_down] = -1
    return sig

def strat_boll_rsi(df):
    sig = pd.Series(0, index=df.index)
    sig.loc[(df['close'] < df['boll_lb']) & (df['rsi'] < 40)] = 1
    sig.loc[(df['close'] > df['boll_ub']) | (df['rsi'] > 65)] = -1
    return sig

def strat_trend_filter(df):
    close = df['close']
    sig = pd.Series(0, index=df.index)
    uptrend = close > df['sma50']
    sig.loc[uptrend & (df['rsi'] < 45) & (df['macd'] > df['macd_signal'])] = 1
    sig.loc[~uptrend | (df['rsi'] > 60)] = -1
    return sig

def strat_atr_stop(df):
    close = df['close']
    sig = pd.Series(0, index=df.index)
    position = 0; entry_price = 0.0
    for i in range(50, len(df)):
        price = float(close.iloc[i]); atr = float(df['atr'].iloc[i])
        if position == 0:
            if (float(close.iloc[i]) > float(df['sma50'].iloc[i])) and (float(df['rsi'].iloc[i]) < 50) and (float(df['macd'].iloc[i]) > float(df['macd_signal'].iloc[i])):
                sig.iloc[i] = 1; position = 1; entry_price = price
        elif position == 1:
            stop = entry_price - 2.0 * atr
            if price < stop or float(df['rsi'].iloc[i]) > 70 or float(df['macd'].iloc[i]) < float(df['macd_signal'].iloc[i]):
                sig.iloc[i] = -1; position = 0; entry_price = 0.0
    return sig

combo_strats = {
    '多数投票(>=2)': strat_majority_vote,
    'RSI+MACD共识': strat_consensus,
    '布林+RSI均值回归': strat_boll_rsi,
    '趋势过滤': strat_trend_filter,
    'ATR止损+趋势': strat_atr_stop,
}

print("=" * 80)
print("各股票最佳策略 vs 组合策略 回测对比")
print("=" * 80)
print()
print(f"{'代码':<10}{'名称':<10}{'最佳策略':<16}{'收益':<10}{'夏普':<8}{'最大回撤':<10}{'胜率'}")
print("-" * 80)
for t, data in prev_results.items():
    best = max(data['strats'].items(), key=lambda x: x[1]['total_return'] if 'error' not in x[1] else -9999)
    b = best[1]
    print(f"{t:<10}{data['name']:<10}{best[0]:<16}{b['total_return']:+.1f}%  {b['sharpe']:.2f}    {b['max_drawdown']:.1f}%     {b['win_rate']:.0f}%")

print()
print("=== 组合策略回测（等权配置7只股票）===")
print()
print(f"{'组合策略':<20}{'7股平均收益':<15}{'平均夏普':<12}{'评价'}")
print("-" * 70)

combo_results = {}
for cs_name, cs_func in combo_strats.items():
    total_ret = 0.0; total_sharpe = 0.0; count = 0
    for t, data in all_data.items():
        df = data['df'].copy()
        df = calc_indicators(df)
        cols_needed = ['sma50','rsi','macd','boll_lb','atr','macd_signal']
        df = df.dropna(subset=cols_needed).reset_index(drop=True)
        if len(df) < 100: continue
        try:
            sig = cs_func(df)
            bt = backtest(df, sig)
            total_ret += bt['total_return']
            total_sharpe += bt['sharpe']
            count += 1
        except: pass
    avg_ret = total_ret / max(count, 1)
    avg_sh = total_sharpe / max(count, 1)
    combo_results[cs_name] = {'avg_return': avg_ret, 'avg_sharpe': avg_sh, 'count': count}
    rating = 'S' if avg_ret > 30 else 'A' if avg_ret > 10 else 'B' if avg_ret > 0 else 'C' if avg_ret > -20 else 'D'
    print(f"{cs_name:<20}{avg_ret:+.1f}%          {avg_sh:.2f}         [{rating}]")

print()
print("=" * 80)
print("总结对比")
print("=" * 80)

ind_returns = []
for t, data in prev_results.items():
    best = max(data['strats'].items(), key=lambda x: x[1]['total_return'] if 'error' not in x[1] else -9999)
    ind_returns.append(best[1]['total_return'])

ind_avg = float(np.mean(ind_returns))
combo_avg = float(np.mean([v['avg_return'] for v in combo_results.values()]))

print()
print(f"  单策略最优 平均收益: {ind_avg:+.1f}%")
print(f"  组合策略   平均收益: {combo_avg:+.1f}%")
winner = "单策略最优" if ind_avg > combo_avg else "组合策略"
print(f"  >>> 结论: {winner} 显著胜出 ({abs(ind_avg-combo_avg):+.1f}%)")

print()
print("各股票: 单策略 vs 组合（取该股组合中最佳）")
print(f"{'代码':<10}{'名称':<10}{'单策略最优':<12}{'组合最优':<12}{'胜出'}")
print("-" * 60)
for t, data in all_data.items():
    df = data['df'].copy()
    df = calc_indicators(df)
    cols_needed = ['sma50','rsi','macd','boll_lb','atr','macd_signal']
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    if len(df) < 100: continue

    pr = prev_results.get(t, {})
    ind_best_ret = -9999.0; ind_best_name = ''
    for sname, sdata in pr.get('strats', {}).items():
        if 'error' not in sdata:
            if sdata['total_return'] > ind_best_ret:
                ind_best_ret = sdata['total_return']
                ind_best_name = sname

    combo_best_ret = -9999.0; combo_best_name = ''
    for cs_name, cs_func in combo_strats.items():
        try:
            sig = cs_func(df)
            bt = backtest(df, sig)
            if bt['total_return'] > combo_best_ret:
                combo_best_ret = bt['total_return']
                combo_best_name = cs_name
        except: pass

    winner = "单策略" if ind_best_ret >= combo_best_ret else "组合"
    diff = abs(ind_best_ret - combo_best_ret)
    print(f"{t:<10}{data['name']:<10}{ind_best_ret:+.1f}%      {combo_best_ret:+.1f}%     {winner}({diff:+.1f}%)")

print()
print("=" * 80)
print("各股票最佳策略（含组合）")
print("=" * 80)
print(f"{'代码':<10}{'名称':<10}{'最佳策略':<22}{'收益':<10}{'类型'}")
print("-" * 70)
for t, data in all_data.items():
    df = data['df'].copy()
    df = calc_indicators(df)
    cols_needed = ['sma50','rsi','macd','boll_lb','atr','macd_signal']
    df = df.dropna(subset=cols_needed).reset_index(drop=True)
    if len(df) < 100: continue

    all_strats = {}
    pr = prev_results.get(t, {})
    for sname, sdata in pr.get('strats', {}).items():
        if 'error' not in sdata:
            all_strats[sname] = sdata['total_return']
    for cs_name, cs_func in combo_strats.items():
        try:
            sig = cs_func(df)
            bt = backtest(df, sig)
            all_strats[f"[C] {cs_name}"] = bt['total_return']
        except: pass

    best_name, best_ret = max(all_strats.items(), key=lambda x: x[1])
    strat_type = "组合" if best_name.startswith("[C]") else "单策略"
    print(f"{t:<10}{data['name']:<10}{best_name:<22}{best_ret:+.1f}%   {strat_type}")

print()
print("=== 综合评分汇总 ===")
score = 0; total = 0
for t, data in prev_results.items():
    best = max(data['strats'].items(), key=lambda x: x[1]['total_return'] if 'error' not in x[1] else -9999)
    if 'error' not in best[1]:
        total += 1
        if best[1]['total_return'] > 0: score += 1
print(f"正收益股票数: {score}/{total}")
