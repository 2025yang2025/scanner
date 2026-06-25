import pandas as pd
import yfinance as yf
import requests
import os

# ==============================================================================
# 1. 核心指標計算與資料優化函數
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    """自訂純 Pandas 計算 MACD 函數"""
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
    """終極提取 Close 欄位函數"""
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            return df.xs('Close', axis=1, level=0).squeeze().astype(float)
        if 'Close' in df.columns.get_level_values(1):
            return df.xs('Close', axis=1, level=1).squeeze().astype(float)
    for col in df.columns:
        if str(col).strip().lower() == 'close':
            return df[col].squeeze().astype(float)
    if df.shape[1] >= 4:
        return df.iloc[:, 3].squeeze().astype(float)
    return pd.Series(dtype=float)

# ==============================================================================
# 2. Telegram 訊息發送功能
# ==============================================================================
def send_telegram_message(message):
    """透過 GitHub Secrets 傳送 Telegram 訊息 (防呆加強版)"""
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("⚠️ 錯誤：找不到 TG_BOT_TOKEN 或 TG_CHAT_ID 環境變數！")
        return

    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if bot_token.lower().startswith("bot"):
        bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("📬 Telegram 訊息發送成功！")
        else:
            print(f"❌ TG 發送失敗，錯誤碼: {response.status_code}, 回傳內容: {response.text}")
    except Exception as e:
        print(f"發送 TG 訊息時發生異常: {e}")

# ==============================================================================
# 3. 多週期共振訊號判斷
# ==============================================================================
def get_signals(ticker):
    try:
        # 抓取資料
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty:
            return False

        # 指標計算
        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        # 條件判定
        weekly_bullish = (w_m > w_s) and (w_h > 0)
        daily_bullish = (d_m > 0) and (d_m > d_s)
        daily_above_ma = (d_c > d_ma_val)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
            
    except Exception as e:
        print(f" ❌ 計算 {ticker} 時發生邏輯錯誤: {e}")
    return False

# ==============================================================================
# 4. 主程式
# ==============================================================================
if __name__ == "__main__":
    print("🚀 開始執行三頻共振選股策略 (動態擴大診斷版)...")
    base_tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD"]
    TICKERS = []
    
    try:
        # 💡 加上 User-Agent 偽裝成一般瀏覽器，防止被維基百科阻擋
        url = "https://zh.wikipedia.org/wiki/%E8%87%BA%E7%81%A350%E6%8C%87%E6%95%B8"
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        tables = pd.read_html(req.text)
        
        # 遞迴檢查維基百科裡面的所有表格，看哪一個包含台灣股票代號
        tw_stocks = []
        for i, table in enumerate(tables):
            code_cols = [col for col in table.columns if '代號' in str(col) or '編號' in str(col)]
            if code_cols:
                col_name = code_cols[0]
                tw_stocks = table[col_name].astype(str).str.strip().tolist()
                print(f"📊 成功在維基百科第 {i} 個表格找到成分股名單！")
                break
                
        tw_tickers = [f"{stock}.TW" for stock in tw_stocks if stock.isdigit()]
        TICKERS = list(set(base_tickers + tw_tickers))
        
    except Exception as e:
        print(f"⚠️ 自動獲取台灣50名單失敗，原因: {e}")
        
    # 如果上面爬蟲失敗，確保有名單可以用
    if not TICKERS:
        print("倒回預設核心股票清單...")
        TICKERS = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW"] + base_tickers

    print(f"📦 本次預計掃描總標的數: {len(TICKERS)} 檔。正在進入計算程序...")

    selected_stocks = []
    for idx, ticker in enumerate(TICKERS, 1):
        # 每計算 10 檔就印一次進度，讓你知道程式沒死掉
        if idx % 10 == 0 or idx == len(TICKERS):
            print(f"⏳ 進度通知: 正在掃描第 {idx}/{len(TICKERS)} 檔股票 ({ticker})...")
            
        if get_signals(ticker):
            print(f"✨ 🎯 發現符合條件標的: {ticker} 🎯 ✨")
            selected_stocks.append(ticker)
            
    # 建立報告訊息
    tw_time = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')
    tg_msg = f"📊 *三頻共振選股報告 (擴大掃描版)*\n⏰ 時間: {tw_time}\n🔍 總掃描標的數: {len(TICKERS)} 檔\n"
    
    if selected_stocks:
        tg_msg += "\n🎯 *今日符合條件標的：*\n"
        for stock in selected_stocks:
            tg_msg += f"• `{stock}`\n"
    else:
        tg_msg += "\n今日無符合多週期共振條件的股票。 💤"
        
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(f"# {tg_msg}")
        
    print("🏁 掃描結束，正在嘗試將報告發送至 Telegram...")
    send_telegram_message(tg_msg)
    print("✅ 全套流程執行完畢！")
