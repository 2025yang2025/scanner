import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime

# ==============================================================================
# 🎯 靜態加載：台灣 50 成分股代碼與中文名稱對照表
# ==============================================================================
STOCK_NAMES = {
    "1101.TW": "台泥", "1102.TW": "亞泥", "1216.TW": "統一", "1301.TW": "台塑",
    "1303.TW": "南亞", "1402.TW": "遠東新", "2002.TW": "中鋼", "2049.TW": "上銀",
    "2105.TW": "正新", "2207.TW": "和泰車", "2303.TW": "聯電", "2308.TW": "台達電",
    "2317.TW": "鴻海", "2324.TW": "仁寶", "2330.TW": "台積電", "2345.TW": "智邦",
    "2352.TW": "佳世達", "2353.TW": "宏碁", "2356.TW": "英業達", "2357.TW": "華碩",
    "2382.TW": "廣達", "2395.TW": "研華", "2408.TW": "南亞科", "2409.TW": "友達",
    "2454.TW": "聯發科", "2498.TW": "宏達電", "2603.TW": "長榮", "2609.TW": "陽明",
    "2615.TW": "萬海", "2881.TW": "富邦金", "2882.TW": "國泰金", "2884.TW": "玉山金",
    "2886.TW": "兆豐金", "2891.TW": "中信金", "2891.TW": "中信金", "2892.TW": "第一金", 
    "2912.TW": "統一超", "3017.TW": "奇鋐", "3034.TW": "聯詠", "3035.TW": "智原", 
    "3045.TW": "台灣大", "3231.TW": "緯創", "3443.TW": "創意", "3481.tw": "群創", 
    "3661.TW": "世芯-KY", "3711.TW": "日月光投控", "4938.TW": "和碩", "5871.TW": "中租-KY", 
    "6116.TW": "彩晶", "6230.TW": "超微體", "6415.TW": "力旺", "9904.TW": "寶成",
    "2395.TW": "研華", "3045.TW": "台灣大", "4938.TW": "和碩"
}

TICKERS = sorted(list(STOCK_NAMES.keys()))

# ==============================================================================
# 1. 核心指標計算與資料優化函數
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def extract_close_series(df):
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
        df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        df_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty:
            return False

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
        pass
    return False

# ==============================================================================
# 4. 主程式
# ==============================================================================
if __name__ == "__main__":
    # 判斷目前時間（台北時間）
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    current_hour = now_tw.hour

    # 💡 判斷是否為早上 08:30 開盤前提醒時段 (上午 11 點以前都認定為早晨提醒)
    if current_hour < 11:
        print("🌅 偵測到目前為早晨開盤前時段，正在讀取昨日報告存檔...")
        if os.path.exists("results.md"):
            with open("results.md", "r", encoding="utf-8") as f:
                saved_content = f.read()
            
            # 將標題替換為開盤前提醒
            remind_msg = saved_content.replace("# 📊 *三頻共振選股報告 (台灣50精準版)*", "🔔 *【開盤前提醒】三頻共振選股報告*")
            print("🏁 成功讀取存檔，正在發送開盤前提醒至 Telegram...")
            send_telegram_message(remind_msg)
            print("✅ 晨間提醒流程執行完畢！")
        else:
            print("⚠️ 找不到 results.md 歷史報告存檔，無法發送晨間提醒。")
        
        # 結束程式，早上不重新運算選股
        exit(0)

    # 📊 下午盤後時段：執行完整的選股運算
    print("🚀 開始執行三頻共振選股策略 (台灣 50 精準版)...")
    print(f"📦 本次預計掃描總標的數: {len(TICKERS)} 檔台灣50權值股。正在進入計算程序...")

    selected_stocks = []
    for idx, ticker in enumerate(TICKERS, 1):
        if idx % 10 == 0 or idx == len(TICKERS):
            print(f"⏳ 進度通知: 正在掃描第 {idx}/{len(TICKERS)} 檔股票 ({ticker})...")
            
        if get_signals(ticker):
            name_zh = STOCK_NAMES.get(ticker, "")
            print(f"✨ 🎯 發現符合條件標的: {ticker} ({name_zh}) 🎯 ✨")
            selected_stocks.append(ticker)
            
    # 建立報告訊息
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')
    tg_msg = f"📊 *三頻共振選股報告 (台灣50精準版)*\n⏰ 時間: {tw_time_str}\n🔍 總掃描標的數: {len(TICKERS)} 檔\n"
    
    if selected_stocks:
        tg_msg += "\n🎯 *今日符合條件標的：*\n"
        for stock in selected_stocks:
            name_zh = STOCK_NAMES.get(stock, "")
            # 💡 這裡將名稱完美格式化為中文對照（例如：2886.TW (兆豐金)）
            tg_msg += f"• `{stock}` (*{name_zh}*)\n"
    else:
        tg_msg += "\n今日無符合多週期共振條件的股票。 💤"
        
    # 寫入 results.md，供隔天早上讀取
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(f"# {tg_msg}")
        
    print("🏁 掃描結束，正在嘗試將報告發送至 Telegram...")
    send_telegram_message(tg_msg)
    print("✅ 下午盤後流程執行完畢！")
