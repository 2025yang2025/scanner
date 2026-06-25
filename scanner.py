import pandas as pd
import yfinance as yf
import requests
import os

# ==============================================================================
# 1. 核心指標計算與資料優化函數
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    """自訂純 Pandas 計算 MACD 函數，避免外掛套件版本衝突"""
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
    """終極提取 Close 欄位函數：無視 yfinance 的各種多層 MultiIndex 結構"""
    if df.empty:
        return pd.Series(dtype=float)
        
    # 情況 1：如果是 MultiIndex (雙層欄位)，嘗試用跨截面提取 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            return df.xs('Close', axis=1, level=0).squeeze().astype(float)
        if 'Close' in df.columns.get_level_values(1):
            return df.xs('Close', axis=1, level=1).squeeze().astype(float)
            
    # 情況 2：如果是單層欄位，尋找欄位名稱包含 'close' (不分大小寫) 的那一欄
    for col in df.columns:
        if str(col).strip().lower() == 'close':
            return df[col].squeeze().astype(float)
            
    # 情況 3：萬一真的找不到，直接撈第 4 欄 (yfinance 預設順序通常是 Open, High, Low, Close)
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
        print("⚠️ 找不到 TG_BOT_TOKEN 或 TG_CHAT_ID，跳過 Telegram 發送。")
        return

    # 防呆機制：去除前後空格，並確保 token 開頭不重複帶有 'bot'
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    
    if bot_token.lower().startswith("bot"):
        bot_token = bot_token[3:] # 移除前三個字元 'bot'

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
        # 抓取不同週期的資料
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        # 提取乾淨的收盤價 Series
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty:
            return False

        # 計算各週期指標
        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        # 提取最新與次新資料的單一數值 (純量)
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

        # 策略條件判定
        # 週線條件：快線 > 慢線 且 柱狀體 > 0 (週線波段多頭)
        weekly_bullish = (w_m > w_s) and (w_h > 0)
        
        # 日線條件：快線 > 0 軸 且 快線 > 慢線 (日線處於多頭主升結構)
        daily_bullish = (d_m > 0) and (d_m > d_s)
        # 日線濾網：股價在日線月線之上
        daily_above_ma = (d_c > d_ma_val)
        
        # 60m條件：快線 > 0 軸 且 柱狀體剛好由負翻正 (當前 > 0, 前一根 <= 0)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        # 全部符合則觸發
        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
            
    except Exception as e:
        # 針對個別股票下載失敗時略過，不中斷整個選股流程
        pass
    return False

#
