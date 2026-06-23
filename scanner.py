import pandas as pd
import yfinance as yf

# 你可以隨時調整這裡的股票代碼
TICKERS = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW"]

# 自訂純 Pandas 計算 MACD 函數
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
    """
    終極提取 Close 欄位函數：無視 yfinance 的各種多層 MultiIndex 結構
    """
    if df.empty:
        return pd.Series(dtype=float)
        
    # 情況 1：如果是 MultiIndex (雙層欄位)，嘗試用跨截面提取 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            return df.xs('Close', axis=1, level=0).squeeze().astype(float)
        if 'Close' in df.columns.get_level_values(1):
            return df.xs('Close', axis=1, level=1).squeeze().astype(float)
            
    # 情況 2：如果是單層欄位，但名稱可能帶有空白或大小寫不一
    # 尋找欄位名稱包含 'close' (不分大小寫) 的那一欄
    for col in df.columns:
        if str(col).strip().lower() == 'close':
            return df[col].squeeze().astype(float)
            
    # 情況 3：萬一真的找不到，直接撈第 4 欄 (yfinance 預設順序通常是 Open, High, Low, Close)
    if df.shape[1] >= 4:
        return df.iloc[:, 3].squeeze().astype(float)
        
    return pd.Series(dtype=float)

def get_signals(ticker):
    try:
        # 1. 抓取不同週期的資料
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        # 2. 透過終極函數提取乾淨的收盤價 Series
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty:
            print(f"⚠️ {ticker} 抓取到的資料結構不完整，跳過")
            return False

        # 3. 計算各週期指標
        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        # 4. 提取最新與次新資料的單一數值 (純量)
        w_m = float(w_macd.iloc[-1])
        w_s = float(w_signal.iloc[-1])
        w_h = float(w_hist.iloc[-1])
        
        d_m = float(d_macd.iloc[-1])
        d_s = float(d_signal.iloc[-1])
        d_c = float(c_daily.iloc[-1])
        d_ma_val = float(d_ma.iloc[-1])
        
        m60_m = float(m60_macd.iloc[-1])
        m60_h = float(m60_hist.iloc[-1])
        m60_h_prev = float(m60_hist.iloc[-2])

        # 5. 策略條件判定
        # 週線條件：快線 > 慢線 且 柱狀體 > 0
        weekly_bullish = (w_m > w_s) and (w_h > 0)
        
        # 日線條件：快線 > 0 軸 且 快線 > 慢線
        daily_bullish = (d_m > 0) and (d_m > d_s)
        # 日線濾網：股價在日線月線之上
        daily_above_ma = (d_c > d_ma_val)
        
        # 60m條件：快線 > 0 軸 且 柱狀體剛好由負翻正 (當前 > 0, 前一根 <= 0)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        # 全部符合則觸發
        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
            
    except Exception as e:
        print(f"計算 {ticker} 時出錯: {e}")
    return False

if __name__ == "__main__":
    print("🚀 開始執行三頻共振選股策略 (無懼改版終極防禦版)...")
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
