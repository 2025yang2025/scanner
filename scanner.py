import pandas as pd
import yfinance as yf
import pandas_ta as ta

# 1. 定義你要監控的股票清單 (以台股權值股為例，可自定義)
TICKERS = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW"]

def get_signals(ticker):
    try:
        # --- 抓取不同週期的資料 ---
        # 60分鐘線 (yfinance 限制最多只能抓最近 730 天的 1h 資料)
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        # 日線
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        # 週線
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        if df_60m.empty or df_daily.empty or df_weekly.empty:
            return False

        # --- 計算指標 ---
        # 週線 MACD
        w_macd = df_weekly.ta.macd(fast=12, slow=26, signal=9)
        # 日線 MACD & 20MA
        d_macd = df_daily.ta.macd(fast=12, slow=26, signal=9)
        d_ma = df_daily.ta.sma(length=20)
        # 60m MACD
        m60_macd = df_60m.ta.macd(fast=12, slow=26, signal=9)

        # 提取最新一筆與上一筆數據 (確保欄位名稱正確，pandas-ta 預設名稱如下)
        # MACD_12_26_9 (快線), MACDs_12_26_9 (慢線), MACDh_12_26_9 (柱狀體)
        w_last = w_macd.iloc[-1]
        
        d_last = d_macd.iloc[-1]
        d_close = df_daily['Close'].iloc[-1]
        d_ma_last = d_ma.iloc[-1]
        
        m60_last = m60_macd.iloc[-1]
        m60_prev = m60_macd.iloc[-2]

        # --- 多週期條件判定 ---
        # 1. 週線：快線 > 慢線 且 柱狀體 > 0
        weekly_bullish = (w_last['MACD_12_26_9'] > w_last['MACDs_12_26_9']) and (w_last['MACDh_12_26_9'] > 0)
        
        # 2. 日線：快線 > 0 且 快線 > 慢線 且 股價在月線上
        daily_bullish = (d_last['MACD_12_26_9'] > 0) and (d_last['MACD_12_26_9'] > d_last['MACDs_12_26_9'])
        daily_above_ma = (d_close > d_ma_last)
        
        # 3. 60m線：快線 > 0 且 柱狀體剛好由負翻正 (當前>0, 前一根<=0)
        m60_cross_up = (m60_last['MACD_12_26_9'] > 0) and (m60_last['MACDh_12_26_9'] > 0) and (m60_prev['MACDh_12_26_9'] <= 0)

        # 綜合判定
        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
    except Exception as e:
        print(f"計算 {ticker} 時出錯: {e}")
    return False

if __name__ == "__main__":
    print("🚀 開始執行三頻共振選股策略...")
    selected_stocks = []
    
    for ticker in TICKERS:
        if get_signals(ticker):
            print(print(f"✨ 🎯 股票 {ticker} 符合篩選條件！"))
            selected_stocks.append(ticker)
            
    # 將結果寫入檔案，供 GitHub 更新使用
    with open("results.md", "w", encoding="utf-8") as f:
        f.write("# 📈 三頻共振選股結果\n")
        f.write(f"更新時間 (UTC): {pd.Timestamp.now()}\n\n")
        if selected_stocks:
            f.write("### 🎯 今日符合條件標的：\n")
            for stock in selected_stocks:
                f.write(f"- **{stock}**\n")
        else:
            f.write("今日無符合條件的股票。\n")
    print("✅ 選股完成，結果已寫入 results.md")
