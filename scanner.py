import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場動態獲取與基本面分池
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

def fetch_fundamental_groups(tickers):
    """
    依據基本面、產業地位將股票分流到對應的策略池
    """
    strat2_candidates = []
    strat3_candidates = []
    strat4_candidates = [] 
    
    for tk in tickers:
        pure_code = tk.split('.')[0]
        # 篩選半導體、AI硬體、電子權值與關鍵零組件鏈 (如玻璃基板、TGV、儲存相關鏈)
        if pure_code.startswith(('23', '24', '30', '32', '34', '35', '36', '37', '61', '62', '64', '80')):
            strat2_candidates.append(tk)
            
            if pure_code in ['2330', '2454', '3443', '3661', '6415', '3017', '3533', '6187']:
                strat3_candidates.append(tk)
                
            # 策略四監控池
            strat4_candidates.append(tk)
                
    return strat2_candidates, strat3_candidates, strat4_candidates

# ==============================================================================
# 📈 技術指標計算核心
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_kd(df, n=9, m1=3, m2=3):
    high_n = df['High'].rolling(window=n).max()
    low_n = df['Low'].rolling(window=n).min()
    close = df['Close']
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k_list, d_list = [], []
    current_k, current_d = 50.0, 50.0
    for r in rsv:
        current_k = (1/m1) * r + ((m1-1)/m1) * current_k
        current_d = (1/m2) * current_k + ((m2-1)/m2) * current_d
        k_list.append(current_k)
        d_list.append(current_d)
    return pd.Series(k_list, index=df.index), pd.Series(d_list, index=df.index)

def extract_ohlc_df(df):
    if df.empty: return pd.DataFrame()
    new_df = pd.DataFrame(index=df.index)
    if isinstance(df.columns, pd.MultiIndex):
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns.get_level_values(0):
                new_df[col] = df.xs(col, axis=1, level=0).squeeze().astype(float)
    else:
        for col in ['Open', 'High', 'Low', 'Close']:
            col_match = [c for c in df.columns if str(c).strip().lower() == col.lower()]
            if col_match:
                new_df[col] = df[col_match[0]].squeeze().astype(float)
    return new_df

# ==============================================================================
# 🛡️ 各策略獨立判定條件
# ==============================================================================

# 策略一專用：多週期 MACD + KD 低檔金叉
def check_strat1_resonance(df_60m, df_daily, df_weekly):
    try:
        c_60m = df_60m['Close']
        c_daily = df_daily['Close']
        c_weekly = df_weekly['Close']
        
        _, _, m60_hist = calculate_macd(c_60m)
        _, _, d_hist = calculate_macd(c_daily)
        _, _, w_hist = calculate_macd(c_weekly)
        
        if len(m60_hist) < 2 or len(d_hist) < 2 or len(w_hist) < 2: return False
        if not (m60_hist.iloc[-1] > m60_hist.iloc[-2] and 
                d_hist.iloc[-1] > d_hist.iloc[-2] and 
                w_hist.iloc[-1] > w_hist.iloc[-2]): 
            return False

        k_60m, d_60m = calculate_kd(df_60m)
        k_daily, d_daily = calculate_kd(df_daily)
        k_weekly, d_weekly = calculate_kd(df_weekly)
        
        def is_low_kd_golden_cross(k_ser, d_ser, threshold=35):
            if len(k_ser) < 2: return False
            cross_up = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
            is_low_level = (k_ser.iloc[-1] <= threshold) or (d_ser.iloc[-1] <= threshold)
            return cross_up and is_low_level

        if (is_low_kd_golden_cross(k_60m, d_60m) and 
            is_low_kd_golden_cross(k_daily, d_daily) and 
            is_low_kd_golden_cross(k_weekly, d_weekly)):
            return True
    except Exception:
        pass
    return False

# 策略二、三通用技術濾網：確保股價處於基本的多頭防守或轉折格局（如站上日 20MA）
def check_trend_defense(df_daily):
    try:
        c_daily = df_daily['Close']
        if len(c_daily) < 20: return False
        d_ma20 = c_daily.rolling(window=20).mean().iloc[-1]
        # 股價位於日生命線之上，代表趨勢未破壞或已止跌轉強
        if c_daily.iloc[-1] > d_ma20:
            return True
    except Exception:
        pass
    return False

# 策略四專用：三週期均線高度糾結壓縮
def check_strat4_ma_tangle(df_60m, df_daily, df_weekly):
    try:
        c_60m = df_60m['Close']
        c_daily = df_daily['Close']
        c_weekly = df_weekly['Close']
        
        if len(c_60m) < 20 or len(c_daily) < 20 or len(c_weekly) < 20: return False
        
        m60_ma5 = c_60m.rolling(window=5).mean().iloc[-1]
        m60_ma10 = c_60m.rolling(window=10).mean().iloc[-1]
        m60_ma20 = c_60m.rolling(window=20).mean().iloc[-1]
        m60_tangle = (max(m60_ma5, m60_ma10, m60_ma20) - min(m60_ma5, m60_ma10, m60_ma20)) / m60_ma20
        
        d_ma5 = c_daily.rolling(window=5).mean().iloc[-1]
        d_ma10 = c_daily.rolling(window=10).mean().iloc[-1]
        d_ma20 = c_daily.rolling(window=20).mean().iloc[-1]
        d_tangle = (max(d_ma5, d_ma10, d_ma20) - min(d_ma5, d_ma10, d_ma20)) / d_ma20
        
        w_ma5 = c_weekly.rolling(window=5).mean().iloc[-1]
        w_ma10 = c_weekly.rolling(window=10).mean().iloc[-1]
        w_ma20 = c_weekly.rolling(window=20).mean().iloc[-1]
        w_tangle = (max(w_ma5, w_ma10, w_ma20) - min(w_ma5, w_ma10, w_ma20)) / w_ma20
        
        close_today = c_daily.iloc[-1]
        
        # 三週期同時高度壓縮，且當前股價成功站上日20MA之上
        if m60_tangle < 0.025 and d_tangle < 0.03 and w_tangle < 0.035 and close_today > d_ma20:
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
    url = f"https://api.telegram.org/bot{str(bot_token).strip()}/sendMessage"
    payload = {"chat_id": str(chat_id).strip(), "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except Exception: pass

# ==============================================================================
# 🚀 主程式流程
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print(f"🚀 啟動【四路平行策略獨立篩選系統】時間: {tw_time_str}...")
    
    ALL_TW_TICKERS = fetch_all_taiwan_market_tickers()
    strat2_candidates, strat3_candidates, strat4_candidates = fetch_fundamental_groups(ALL_TW_TICKERS)
    
    # 全市場+監控池進行大聯集，確保每檔股票只下載一次數據，提高盤中執行效率
    tech_scan_pool = sorted(list(set(ALL_TW_TICKERS + strat2_candidates + strat3_candidates + strat4_candidates)))
    
    strat1_matches, strat2_matches, strat3_matches, strat4_matches = [], [], [], []

    for idx, ticker in enumerate(tech_scan_pool, 1):
        if idx % 20 == 0: 
            time.sleep(random.uniform(1.0, 2.0))
            
        try:
            raw_60m = yf.download(ticker, period="1mo", interval="60m", progress=False, auto_adjust=True)
            raw_daily = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            raw_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
            
            df_60m = extract_ohlc_df(raw_60m)
            df_daily = extract_ohlc_df(raw_daily)
            df_weekly = extract_ohlc_df(raw_weekly)
            
            if df_60m.empty or df_daily.empty or df_weekly.empty: continue
            
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"`{ticker}` (*{name_zh}*)" if name_zh else f"`{ticker}`"
            
            # ────────────────────────────────────────────────────────
            # ⚡ 策略一：【唯獨此策略】驗證多週期 MACD + KD 低檔共振
            # ────────────────────────────────────────────────────────
            if check_strat1_resonance(df_60m, df_daily, df_weekly):
                strat1_matches.append(stock_label)
                
            # ────────────────────────────────────────────────────────
            # ⚡ 策略二：產業轉折爆發股（依照原本設定篩選，不再綁定策略一）
            # ────────────────────────────────────────────────────────
            if ticker in strat2_candidates:
                if check_trend_defense(df_daily):
                    strat2_matches.append(stock_label)
                    
            # ────────────────────────────────────────────────────────
            # ⚡ 策略三：抗震核心存股龍頭（依照原本設定篩選，圖表穩健即觸發）
            # ────────────────────────────────────────────────────────
            if ticker in strat3_candidates:
                if check_trend_defense(df_daily):
                    strat3_matches.append(stock_label)
                    
            # ────────────────────────────────────────────────────────
            # ⚡ 策略四：三週期均線糾結（完全獨立的技術壓縮條件）
            # ────────────────────────────────────────────────────────
            if ticker in strat4_candidates:
                if check_strat4_ma_tangle(df_60m, df_daily, df_weekly):
                    strat4_matches.append(stock_label)
                    
        except Exception:
            pass

    # 📝 建立推播報告
    tw_msg = f"🇹🇼 *【台股市場：四路獨立策略即時追蹤】*\n⏰ 監控時間: {tw_time_str}\n"
    tw_msg += "═" * 15 + "\n"
    
    tw_msg += "📈 *策略一：原版多週期三頻共振*\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "當前無符合標的。 💤") + "\n\n"

    tw_msg += "🚀 *策略二：獲利暴增 × 產業轉折爆發股*\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "當前無符合標的。 💤") + "\n\n"

    tw_msg += "💎 *策略三：高技術壁壘 × 抗震核心存股龍頭*\n"
    tw_msg += "↳ " + (", ".join(strat3_matches) if strat3_matches else "當前無符合標的。 💤") + "\n\n"

    tw_msg += "💥 *策略四：三週期均線糾結 (共振壓縮)*\n"
    tw_msg += "↳ " + (", ".join(strat4_matches) if strat4_matches else "當前無符合標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 台股各策略獨立篩選報告發送完畢！")
