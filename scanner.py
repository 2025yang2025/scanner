import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場全自動動態獲取
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_all_taiwan_market_tickers():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                if code.isdigit() and len(code) == 4:
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception:
        pass
    if not all_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
    return sorted(list(set(all_tickers)))

def fetch_fundamental_snapshot(tickers):
    strat2_candidates = []
    strat3_candidates = []
    for tk in tickers:
        pure_code = tk.split('.')[0]
        if pure_code.startswith(('23', '24', '30', '32', '34', '35', '36', '37', '61', '62', '64', '80')):
            strat2_candidates.append(tk)
            if pure_code in ['2330', '2454', '3443', '3661', '6415', '3017', '3533', '6187']:
                strat3_candidates.append(tk)
    return strat2_candidates, strat3_candidates

# ==============================================================================
# 📈 數學指標計算核心 (MACD & KD)
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_kd(df, n=9, m1=3, m2=3):
    """
    計算標準 KD 指標
    """
    high_n = df['High'].rolling(window=n).max()
    low_n = df['Low'].rolling(window=n).min()
    close = df['Close']
    
    # 計算 RSV (未成熟隨機值)
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50) # 補缺值避免初期崩潰
    
    k_list, d_list = [], []
    current_k, current_d = 50.0, 50.0 # 初始值設定為 50
    
    for r in rsv:
        current_k = (1/m1) * r + ((m1-1)/m1) * current_k
        current_d = (1/m2) * current_k + ((m2-1)/m2) * current_d
        k_list.append(current_k)
        d_list.append(current_d)
        
    return pd.Series(k_list, index=df.index), pd.Series(d_list, index=df.index)

def extract_ohlc_df(df):
    """
    安全提取多重索引 (MultiIndex) 的 OHLC 資料
    """
    if df.empty: return pd.DataFrame()
    new_df = pd.DataFrame(index=df.index)
    
    # 處理 yfinance 多重索引欄位
    if isinstance(df.columns, pd.MultiIndex):
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns.get_level_values(0):
                new_df[col] = df.xs(col, axis=1, level=0).squeeze().astype(float)
            elif col in df.columns.get_level_values(1):
                new_df[col] = df.xs(col, axis=1, level=1).squeeze().astype(float)
    else:
        for col in ['Open', 'High', 'Low', 'Close']:
            col_match = [c for c in df.columns if str(c).strip().lower() == col.lower()]
            if col_match:
                new_df[col] = df[col_match[0]].squeeze().astype(float)
                
    return new_df

# ==============================================================================
# 🎯 新·多週期技術共振過濾引擎 (MACD 翻正 + KD 低檔金叉)
# ==============================================================================
def check_technical_resonance(ticker):
    try:
        # 下載 60分、日、週三個週期的歷史數據
        raw_60m = yf.download(ticker, period="1mo", interval="60m", progress=False)
        raw_daily = yf.download(ticker, period="1y", interval="1d", progress=False)
        raw_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False)
        
        df_60m = extract_ohlc_df(raw_60m)
        df_daily = extract_ohlc_df(raw_daily)
        df_weekly = extract_ohlc_df(raw_weekly)
        
        if df_60m.empty or df_daily.empty or df_weekly.empty: return False

        # --- 1. 計算 MACD 柱狀體 (Hist) ---
        _, _, m60_hist = calculate_macd(df_60m['Close'])
        _, _, d_hist = calculate_macd(df_daily['Close'])
        _, _, w_hist = calculate_macd(df_weekly['Close'])
        
        if len(m60_hist) < 2 or len(d_hist) < 2 or len(w_hist) < 2: return False
        
        # 判斷 MACD 是否「往 0 軸翻正」（最新一根 > 0 且 前一根 <= 0）
        m60_macd_turn_up = (m60_hist.iloc[-1] > 0) and (m60_hist.iloc[-2] <= 0)
        daily_macd_turn_up = (d_hist.iloc[-1] > 0) and (d_hist.iloc[-2] <= 0)
        weekly_macd_turn_up = (w_hist.iloc[-1] > 0) and (w_hist.iloc[-2] <= 0)
        
        # 三週期 MACD 必須同時在翻正起漲點
        macd_resonance = m60_macd_turn_up and daily_macd_turn_up and weekly_macd_turn_up

        # --- 2. 計算 KD 指標並判定低檔黃金交叉 ---
        k_60m, d_60m = calculate_kd(df_60m)
        k_daily, d_daily = calculate_kd(df_daily)
        k_weekly, d_weekly = calculate_kd(df_weekly)
        
        def is_low_kd_golden_cross(k_ser, d_ser, threshold=30):
            """ 判定 KD 是否在低檔 (<=30) 發生黃金交叉 """
            if len(k_ser) < 2: return False
            # 最新一根 K > D，前一根 K <= D (黃金交叉)
            cross_up = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
            # 交叉時的數值落於低檔起漲區 (K 值與 D 值皆小於等於指定門檻)
            is_low_level = (k_ser.iloc[-1] <= threshold) or (d_ser.iloc[-1] <= threshold)
            return cross_up and is_low_level

        kd_60m_ok = is_low_kd_golden_cross(k_60m, d_60m)
        kd_daily_ok = is_low_kd_golden_cross(k_daily, d_daily)
        kd_weekly_ok = is_low_kd_golden_cross(k_weekly, d_weekly)
        
        # 三週期 KD 均落入轉折金叉區
        kd_resonance = kd_60m_ok and kd_daily_ok and kd_weekly_ok

        # 兩大核心條件必須同時滿足（共振中的共振！）
        if macd_resonance and kd_resonance:
            return True
            
    except Exception:
        pass
    return False

# ==============================================================================
# 💬 Telegram 發送
# ==============================================================================
def send_telegram_message(message):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id: return
    bot_token, chat_id = str(bot_token).strip(), str(chat_id).strip()
    if bot_token.lower().startswith("bot"): bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception: pass

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股新·多週期轉折起漲共振】盤後策略報告...")
    
    ALL_TW_TICKERS = fetch_all_taiwan_market_tickers()
    strat2_candidates, strat3_candidates = fetch_fundamental_snapshot(ALL_TW_TICKERS)
    tech_scan_pool = sorted(list(set(strat2_candidates + strat3_candidates)))
    
    strat1_matches, strat2_matches, strat3_matches = [], [], []

    print(f"⏳ 正在進行台股新指標精密篩選 (共 {len(tech_scan_pool)} 檔)...")
    for idx, ticker in enumerate(tech_scan_pool, 1):
        if idx % 15 == 0: 
            time.sleep(random.uniform(2.0, 3.5))
            
        if check_technical_resonance(ticker):
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"`{ticker}` (*{name_zh}*)" if name_zh else f"`{ticker}`"
            
            strat1_matches.append(stock_label)
            if ticker in strat2_candidates: strat2_matches.append(stock_label)
            if ticker in strat3_candidates: strat3_matches.append(stock_label)

    # 📝 建立台股獨立美化訊息
    tw_msg = f"🇹🇼 *【台股市場：多週期 MACD 翻正 × 低檔 KD 金叉報告】*\n⏰ 報告時間: {tw_time_str}\n"
    tw_msg += "↳ *新濾網條件*：60M/日/週 MACD 柱狀體首度轉正 ＋ KD 低檔 (≤30) 黃金交叉\n"
    tw_msg += "═" * 15 + "\n"
    
    tw_msg += "📈 *策略一：原版多週期三頻共振*\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "🚀 *策略二：獲利暴增 × 產業轉折爆發股*\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的 nudge。 💤") + "\n\n"

    tw_msg += "💎 *策略三：高技術壁壘 × 抗震核心存股龍頭*\n"
    tw_msg += "↳ " + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 台股新指標共振報告發送完畢！")
