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
            if buy_shares > 0: cash -= float(buy_shares)*price; shares += buy_shares; trades.append({'type':'BUY','price':price,'date':df['date'].iloc[i]})
        elif sig == -1 and shares > 0:
            sell_shares = int(shares * 0.95)
            if sell_shares > 0: cash += float(sell_shares)*price; shares -= sell_shares; trades.append({'type':'SELL','price':price,'date':df['date'].iloc[i]})
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
    win_rate = 0.0; avg_hold_days = 0.0; trade_returns = []
    if len(closed) >= 1:
        buy_p = float(initial_capital); wins = []; losses = []; hold_days_list = []; prev_buy_date = None
        for t in closed:
            ret = (float(t['price']) - buy_p) / buy_p * 100.0
            trade_returns.append(ret)
            if ret > 0: wins.append(ret)
            else: losses.append(ret)
            buy_p = float(t['price'])
        denom = max(len(wins)+len(losses),1)
        win_rate = float(len(wins)) / denom * 100.0
    return {'total_return':float(total_return),'sharpe':float(sharpe),'max_drawdown':float(max_dd)*100.0,'num_trades':len(closed),'win_rate':float(win_rate),'equity_curve':equity_curve,'trades':trades}

def strat_macd_cross(df):
    sig = pd.Series(0, index=df.index)
    sig.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())] = 1
    sig.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())] = -1
    return sig

def strat_boll(df):
    sig = pd.Series(0, index=df.index)
    sig.loc[df['close'] < df['boll_lb']] = 1; sig.loc[df['close'] > df['boll_ub']] = -1
    return sig

def strat_ma_cross(df):
    close = df['close']
    sig = pd.Series(0, index=df.index)
    sig.loc[(close > df['sma50']) & (close.shift() <= df['sma50'].shift())] = 1
    sig.loc[(close < df['sma50']) & (close.shift() >= df['sma50'].shift())] = -1
    return sig

def strat_trend(df):
    close = df['close']
    sig = pd.Series(0, index=df.index)
    trend_on = (close > df['sma50']) & (df['rsi'] > 50) & (df['macd'] > 0)
    sig.loc[trend_on] = 1; sig.loc[~trend_on] = -1
    return sig

def strat_rsi_macd(df):
    sig = pd.Series(0, index=df.index)
    macd_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
    macd_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
    sig.loc[(df['rsi'] < 40) & macd_up] = 1
    sig.loc[(df['rsi'] > 60) | macd_down] = -1
    return sig

def strat_rsi(df):
    sig = pd.Series(0, index=df.index)
    sig.loc[df['rsi'] < 30] = 1; sig.loc[df['rsi'] > 70] = -1
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

strat_map = {
    'MACD交叉': strat_macd_cross,
    '布林带': strat_boll,
    '均线交叉': strat_ma_cross,
    '趋势追踪': strat_trend,
    'RSI+MACD': strat_rsi_macd,
    'RSI(30/70)': strat_rsi,
}

# Compute per-year returns for key stocks
print("=" * 80)
print("分年度收益分析（最近5年）")
print("=" * 80)

for t, data in all_data.items():
    df = data['df'].copy()
    df = calc_indicators(df)
    df = df.dropna(subset=['sma50','rsi','macd','boll_lb','atr','macd_signal']).reset_index(drop=True)
    
    # Best strategy for this stock
    pr = prev_results.get(t, {})
    best_name = max(pr.get('strats', {}).items(), key=lambda x: x[1]['total_return'] if 'error' not in x[1] else -9999)[0] if pr.get('strats') else None
    
    if best_name and best_name in strat_map:
        strat_func = strat_map[best_name]
        sig = strat_func(df)
        bt = backtest(df, sig)
        
        # Compute yearly returns
        df['year'] = df['date'].dt.year
        yearly_returns = []
        for year in sorted(df['year'].unique())[-6:]:  # last 6 years
            yr_df = df[df['year'] == year].copy()
            yr_df['signal'] = strat_func(yr_df)
            yr_bt = backtest(yr_df, yr_df['signal'])
            yearly_returns.append((year, yr_bt['total_return']))
        
        print(f"\n{t} {data['name']} - 最优策略:{best_name} (总体{bt['total_return']:+.1f}%)")
        for yr, ret in yearly_returns:
            bar = '+' * int(max(0, ret)/5) if ret > 0 else '-' * int(max(0, -ret)/5)
            print(f"  {yr}: {ret:+7.1f}%  {bar}")
    else:
        print(f"\n{t} {data['name']} - 无最优策略数据")

print()
print("=" * 80)
print("风险收益详细对比")
print("=" * 80)

# Per stock: compare best 2 strategies side by side
for t, data in all_data.items():
    df = data['df'].copy()
    df = calc_indicators(df)
    df = df.dropna(subset=['sma50','rsi','macd','boll_lb','atr','macd_signal']).reset_index(drop=True)
    
    pr = prev_results.get(t, {})
    sorted_strats = sorted([(k,v) for k,v in pr.get('strats',{}).items() if 'error' not in v], 
                           key=lambda x: x[1]['total_return'], reverse=True)
    
    if len(sorted_strats) >= 2:
        best = sorted_strats[0]
        second = sorted_strats[1]
        print(f"\n{t} {data['name']}")
        print(f"  第1: {best[0]:<12} 收益{best[1]['total_return']:+.1f}%  夏普{best[1]['sharpe']:.2f}  回撤{best[1]['max_drawdown']:.1f}%  交易{best[1]['num_trades']}次  胜率{best[1]['win_rate']:.0f}%")
        print(f"  第2: {second[0]:<12} 收益{second[1]['total_return']:+.1f}%  夏普{second[1]['sharpe']:.2f}  回撤{second[1]['max_drawdown']:.1f}%  交易{second[1]['num_trades']}次  胜率{second[1]['win_rate']:.0f}%")
        
        # Risk-adjusted comparison
        risk_adj_1 = best[1]['total_return'] / max(best[1]['max_drawdown'], 1)
        risk_adj_2 = second[1]['total_return'] / max(second[1]['max_drawdown'], 1)
        print(f"  风险收益比: {risk_adj_1:.2f} vs {risk_adj_2:.2f}  ({'第1更优' if risk_adj_1 > risk_adj_2 else '第2更优'})")
