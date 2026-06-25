import pandas as pd
import yfinance as yf
import requests
import os

TICKERS = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW"]

def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0): return df.xs('Close', axis=1, level=0).squeeze().astype(float)
        if 'Close' in df.columns.get_level_values(1): return df.xs('Close', axis=1, level=1).squeeze().astype(float)
    for col in df.columns:
        if str(col).strip().lower() == 'close': return df[col].squeeze().astype(float)
    if df.shape[1] >= 4: return df.iloc[:, 3].squeeze().astype(float)
    return pd.Series(dtype=float)

def send_telegram_message(message):
    """透過 GitHub Secrets 傳送 Telegram 訊息"""
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("⚠️ 找不到 TG_BOT_TOKEN 或 TG_CHAT_ID，跳過 Telegram 發送。")
        return

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

def get_signals(ticker):
    try:
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty: return False

        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        weekly_bullish = (w_m > w_s) and (w_h > 0)
        daily_bullish = (d_m > 0) and (d_m > d_s)
        daily_above_ma = (d_c > d_ma_val)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
            return True
            
    except Exception as e:
        print(f"計算 {ticker} 時出錯: {e}")
    return False

if __name__ == "__main__":
    print("🚀 開始執行三頻共振選股策略 (自動通知版)...")
    selected_stocks = []
    
    for ticker in TICKERS:
        if get_signals(ticker):
            print(f"✨ 🎯 {ticker} 符合條件")
            selected_stocks.append(ticker)
            
    # 準備訊息內容
    tw_time = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')
    
    tg_msg = f"📊 *三頻共振選股報告*\n⏰ 時間: {tw_time}\n"
    if selected_stocks:
        tg_msg += "🎯 *今日符合條件標的：*\n"
        for stock in selected_stocks:
            tg_msg += f"• `{stock}`\n"
    else:
        tg_msg += "今日無符合多週期共振條件的股票。 💤"
        
    # 同步寫入 results.md 檔案
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(f"# {tg_msg}")
        
    # 🚀 發送訊息到 Telegram
    send_telegram_message(tg_msg)
    print("✅ 選股與發送流程全部完成")
