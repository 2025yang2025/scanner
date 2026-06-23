import pandas as pd
import yfinance as yf

# 你可以隨時調整這裡的股票代碼
TICKERS = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW"]

# 自訂純 Pandas 計算 MACD 函數
def calculate_macd(df, fast=12, slow=26, signal=9):
    fast_ema = df['Close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def get_signals(ticker):
    try:
        # 抓取不同週期的資料
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        if df_60m.empty or df_daily.empty or df_weekly.empty:
            return False

        # 1. 計算週線 MACD
        w_macd, w_signal, w_hist = calculate_macd(df_weekly)
        # 2. 計算日線 MACD 及 20MA (月線)
        d_macd, d_signal, d_hist = calculate_macd(df_daily)
        d_ma = df_daily['Close'].rolling(window=20).mean()
        # 3. 計算 60m MACD
        m60_macd, m60_signal, m60_hist = calculate_macd(df_60m)

        # --- 策略條件判定 ---
        # 週線條件：快線 > 慢線 且 柱狀體 > 0
        weekly_bullish = (w_macd.iloc[-1] > w_signal.iloc[-1]) and (w_hist.iloc[-1] > 0)
        
        # 日線條件：快線 > 0 軸 且 快線 > 慢線
        daily_bullish = (d_macd.iloc[-1] > 0) and (d_macd.iloc[-1] > d_signal.iloc[-1])
        # 日線濾網：股價在日線月線之上
        daily_above_ma = (df_daily['Close'].iloc[-1] > d_ma.iloc[-1])
        
        # 60m條件：快線 > 0 軸 且 柱狀體剛好由負翻正 (當前 > 0, 前一根 <= 0)
        m60_cross_up = (m60_macd.iloc[-1] > 0) and (m60_hist.iloc[-1] > 0) and (m60_hist.iloc[-2] <= 0)

        # 全部符合則觸發
        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
            
    except Exception as e:
        print(f"計算 {ticker} 時出錯: {e}")
    return False

if __name__ == "__main__":
    print("🚀 開始執行三頻共振選股策略 (純淨版)...")
    selected_stocks = []
    
    for ticker in TICKERS:
        if get_signals(ticker):
            print(f"✨ 🎯 股票 {ticker} 符合篩選條件！")
            selected_stocks.append(ticker)
            
    # 將選股結果寫入 Markdown
    with open("results.md", "w", encoding="utf-8") as f:
        f.write("# 📈 三頻共振選股結果\n")
        tw_time = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"更新時間 (台北時間): {tw_time}\n\n")
        if selected_stocks:
            f.write("### 🎯 今日符合條件標的：\n")
            for stock in selected_stocks:
                f.write(f"- **{stock}**\n")
        else:
            f.write("今日無符合條件的股票。\n")
            
    print("✅ 選股完成")
