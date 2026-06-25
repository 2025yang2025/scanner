import pandas as pd
import yfinance as yf
import requests
import os

# ==============================================================================
# 備援防線：當所有網路爬蟲都失敗時的「保底台股名單」與基礎中文對照表
# ==============================================================================
DEFAULT_STOCK_NAMES = {
    "1101.TW": "台泥", "1102.TW": "亞泥", "1216.TW": "統一", "1301.TW": "台塑",
    "1303.TW": "南亞", "1402.TW": "遠東新", "2002.TW": "中鋼", "2049.TW": "上銀",
    "2105.TW": "正新", "2207.TW": "和泰車", "2303.TW": "聯電", "2308.TW": "台達電",
    "2317.TW": "鴻海", "2324.TW": "仁寶", "2330.TW": "台積電", "2345.TW": "智邦",
    "2352.TW": "佳世達", "2353.TW": "宏碁", "2356.TW": "英業達", "2357.TW": "華碩",
    "2382.TW": "廣達", "2395.TW": "研華", "2408.TW": "南亞科", "2409.TW": "友達",
    "2454.TW": "聯發科", "2498.TW": "宏達電", "2603.TW": "長榮", "2609.TW": "陽明",
    "2615.TW": "萬海", "2881.TW": "富邦金", "2882.TW": "國泰金", "2884.TW": "玉山金",
    "2886.TW": "兆豐金", "2891.TW": "中信金", "2892.TW": "第一金", "2912.TW": "統一超",
    "3017.TW": "奇鋐", "3034.TW": "聯詠", "3035.TW": "智原", "3045.TW": "台灣大",
    "3231.TW": "緯創", "3443.TW": "創意", "3481.TW": "群創", "3661.TW": "世芯-KY",
    "3711.TW": "日月光投控", "4938.TW": "和碩", "5871.TW": "中租-KY", "6116.TW": "彩晶",
    "6230.TW": "超微體", "6415.TW": "力旺", "9904.TW": "寶成"
}

# 全域字典：程式運作時會動態塞入最新名稱
DYNAMIC_STOCK_NAMES = {}

# ==============================================================================
# 🌐 全自動多管道成分股獲取函數 (核心亮點)
# ==============================================================================
def fetch_latest_taiwan_50():
    """多管道動態抓取最新的台灣 50 成分股與中文名稱"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # --- 管道一：證交所官方開放資料 API (第一優先) ---
    try:
        print("🌐 嘗試透過【證交所開放資料 API】獲取成分股...")
        url = "https://openapi.twse.com.tw/v1/opendata/t187ap05_P"
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tickers = []
            for item in data:
                # 篩選臺灣50指數成分股
                if "臺灣50指數" in item.get("IndexName", "") or "0050" in item.get("IndexName", ""):
                    code = item.get("StockCode", "").strip()
                    name = item.get("StockName", "").strip()
                    if code.isdigit() and len(code) == 4:
                        tk = f"{code}.TW"
                        tickers.append(tk)
                        DYNAMIC_STOCK_NAMES[tk] = name
            if tickers:
                print(f"✅ 成功！證交所管道動態加載 {len(tickers)} 檔成分股。")
                return sorted(list(set(tickers)))
    except Exception as e:
        print(f"⚠️ 證交所管道失效，原因: {e}")

    # --- 管道二：FinMind 備用 API (加上 Timeout 防卡死) ---
    try:
        print("🌐 嘗試透過【FinMind API】獲取成分股...")
        fm_url = "https://api.finmindapi.com/v4/data?dataset=TaiwanStockHoldingSharesPer&data_id=0050"
        res = requests.get(fm_url, headers=headers, timeout=10).json()
        if res.get("status") == 200 and "data" in res:
            tickers = []
            for item in res["data"]:
                code = item["holding_ticker"].strip()
                name = item.get("holding_name", "").strip()
                if code.isdigit() and len(code) == 4:
                    tk = f"{code}.TW"
                    tickers.append(tk)
                    # 如果 API 有給中文名稱就用，沒有就留空
                    DYNAMIC_STOCK_NAMES[tk] = name if name else DEFAULT_STOCK_NAMES.get(tk, "未明個股")
            if tickers:
                print(f"✅ 成功！FinMind 管道動態加載 {len(tickers)} 檔成分股。")
                return sorted(list(set(tickers)))
    except Exception as e:
        print(f"⚠️ FinMind 管道失效，原因: {e}")

    # --- 最終防線：若上述網路全部斷線，啟用保底機制 ---
    print("🚨 警告：所有自動化爬蟲管道均因網路或 DNS 限制失敗！啟動保底防禦機制...")
    for k, v in DEFAULT_STOCK_NAMES.items():
        DYNAMIC_STOCK_NAMES[k] = v
    return sorted(list(DEFAULT_STOCK_NAMES.keys()))

# ==============================================================================
# 指標計算、資料提取與 TG 發送功能 (保持高度穩定結構)
# ==============================================================================
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
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id: return
    bot_token, chat_id = str(bot_token).strip(), str(chat_id).strip()
    if bot_token.lower().startswith("bot"): bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200: print("📬 Telegram 訊息發送成功！")
        else: print(f"❌ TG 發送失敗: {response.text}")
    except Exception as e: print(f"發送 TG 異常: {e}")

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
    except Exception: pass
    return False

# ==============================================================================
# 4. 主程式：自動化流程
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    current_hour = now_tw.hour

    # 🌅 早上時段：直接讀取歷史存檔
    if current_hour < 11:
        print("🌅 偵測到早晨開盤前時段，正在讀取昨日報告存檔...")
        if os.path.exists("results.md"):
            with open("results.md", "r", encoding="utf-8") as f:
                saved_content = f.read()
            remind_msg = saved_content.replace("# 📊 *三頻共振選股報告 (台灣50全自動版)*", "🔔 *【開盤前提醒】三頻共振選股報告*")
            send_telegram_message(remind_msg)
            print("✅ 晨間提醒流程執行完畢！")
        else:
            print("⚠️ 找不到歷史存檔。")
        exit(0)

    # 📊 下午盤後時段：動態抓取與運算
    print("🚀 開始執行三頻共振選股策略 (台灣 50 全自動動態版)...")
    
    # 💡 呼叫多重防禦爬蟲，取得最新成分股
    TICKERS = fetch_latest_taiwan_50()
    print(f"📦 本次實際掃描總標的數: {len(TICKERS)} 檔。正在進入計算程序...")

    selected_stocks = []
    for idx, ticker in enumerate(TICKERS, 1):
        if idx % 10 == 0 or idx == len(TICKERS):
            print(f"⏳ 進度通知: 正在掃描第 {idx}/{len(TICKERS)} 檔股票 ({ticker})...")
            
        if get_signals(ticker):
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "新進榜個股")
            print(f"✨ 🎯 發現符合條件標的: {ticker} ({name_zh}) 🎯 ✨")
            selected_stocks.append(ticker)
            
    # 建立報告訊息
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')
    tg_msg = f"📊 *三頻共振選股報告 (台灣50全自動版)*\n⏰ 時間: {tw_time_str}\n🔍 總掃描標的數: {len(TICKERS)} 檔\n"
    
    if selected_stocks:
        tg_msg += "\n🎯 *今日符合條件標的：*\n"
        for stock in selected_stocks:
            name_zh = DYNAMIC_STOCK_NAMES.get(stock, "新進榜個股")
            tg_msg += f"• `{stock}` (*{name_zh}*)\n"
    else:
        tg_msg += "\n今日無符合多週期共振條件的股票。 💤"
        
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(f"# {tg_msg}")
        
    print("🏁 掃描結束，正在嘗試將報告發送至 Telegram...")
    send_telegram_message(tg_msg)
    print("✅ 下午盤後全自動流程執行完畢！")
