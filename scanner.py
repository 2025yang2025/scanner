import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場技術面模組 (保持全自動含中文功能)
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
                    # 篩選半導體、AI硬體、電子權值與關鍵零組件鏈
                    if code.startswith(('23', '24', '30', '32', '34', '35', '36', '37', '61', '62', '64', '80')):
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

def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_kd(df, n=9, m1=3, m2=3):
    # 確保傳入的 DataFrame 欄位名稱乾淨
    df_clean = df.copy()
    if isinstance(df_clean.columns, pd.MultiIndex):
        df_clean.columns = [col[0] for col in df_clean.columns]
    
    low_min = df_clean['Low'].squeeze().astype(float).rolling(window=n).min()
    high_max = df_clean['High'].squeeze().astype(float).rolling(window=n).max()
    close = df_clean['Close'].squeeze().astype(float)
    
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    rsv = rsv.fillna(50)
    
    k_list, d_list = [50.0], [50.0]
    for i in range(1, len(rsv)):
        current_k = (k_list[-1] * (m1 - 1) + rsv.iloc[i]) / m1
        current_d = (d_list[-1] * (m2 - 1) + current_k) / m2
        k_list.append(current_k)
        d_list.append(current_d)
        
    return pd.Series(k_list, index=df.index), pd.Series(d_list, index=df.index)

def calculate_rsi(close_series, period=6):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def extract_close_series(df):
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0): 
            return df.xs('Close', axis=1, level=0).squeeze().astype(float)
    for col in df.columns:
        if str(col).strip().lower() == 'close': 
            return df[col].squeeze().astype(float)
    return pd.Series(dtype=float)

def extract_column_series(df, col_name):
    if df.empty: return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if col_name in df.columns.get_level_values(0): 
            return df.xs(col_name, axis=1, level=0).squeeze().astype(float)
    for col in df.columns:
        if str(col).strip().lower() == col_name.lower(): 
            return df[col].squeeze().astype(float)
    return pd.Series(dtype=float)

def check_volume_filter(df_daily, min_volume=500):
    try:
        v_daily = extract_column_series(df_daily, 'Volume')
        if len(v_daily) < 20: return False
        v_ma20_shares = v_daily.rolling(window=20).mean().iloc[-1]
        v_ma20_sheets = v_ma20_shares / 1000
        return v_ma20_sheets >= min_volume
    except:
        return False

# ==============================================================================
# 🎯 核心策略檢測邏輯 (傳入已下載之資料，避免重複下載)
# ==============================================================================

def check_strat1_resonance(df_60m, df_daily, df_weekly):
    """ 策略一：原版多週期三頻共振 (MACD) + KD低檔金叉 """
    try:
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if c_60m.empty or c_daily.empty or c_weekly.empty: return False

        # 1. MACD 條件計算
        w_macd, w_signal, w_hist = calculate_macd(c_weekly)
        d_macd, d_signal, d_hist = calculate_macd(c_daily)
        d_ma = c_daily.rolling(window=20).mean()
        m60_macd, m60_signal, m60_hist = calculate_macd(c_60m)

        if len(w_hist) < 1 or len(d_hist) < 1 or len(m60_hist) < 2: return False

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        macd_cond = (w_m > w_s) and (w_h > 0) and (d_m > 0) and (d_m > d_s) and (d_c > d_ma_val) and (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)
        
        if not macd_cond: return False

        # 2. KD 條件計算 (低檔 35 以下金叉)
        k_60m, d_60m = calculate_kd(df_60m)
        k_daily, d_daily = calculate_kd(df_daily)
        k_weekly, d_weekly = calculate_kd(df_weekly)
        
        def is_low_kd_gold(k_ser, d_ser, threshold=35):
            if len(k_ser) < 2: return False
            cross_up = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
            is_low = (k_ser.iloc[-1] <= threshold) or (d_ser.iloc[-1] <= threshold)
            return cross_up and is_low

        if is_low_kd_gold(k_60m, d_60m) and is_low_kd_gold(k_daily, d_daily) and is_low_kd_gold(k_weekly, d_weekly):
            return True
    except Exception:
        pass
    return False

def check_oversold_rebound(df_daily):
    """ 策略二：季線跌深負乖離 × KD金叉 (保持原本設定) """
    try:
        if df_daily.empty or len(df_daily) < 60: return False
        
        c_daily = extract_close_series(df_daily)
        ma60 = c_daily.rolling(window=60).mean().iloc[-1]
        close_today = c_daily.iloc[-1]
        bias_60 = (close_today - ma60) / ma60
        
        k_series, d_series = calculate_kd(df_daily)
        if bias_60 <= -0.15 and k_series.iloc[-1] < 25 and d_series.iloc[-1] < 25:
            if k_series.iloc[-1] > d_series.iloc[-1] and k_series.iloc[-2] <= d_series.iloc[-2]:
                return True
    except Exception:
        pass
    return False

def check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    """ 策略三：60分K/日K/週K同步均線糾結 (保持原本設定) """
    try:
        c_60m = extract_close_series(df_60m)
        c_daily = extract_close_series(df_daily)
        c_weekly = extract_close_series(df_weekly)
        
        if len(c_60m) < 20 or len(c_daily) < 20 or len(c_weekly) < 20: return False
        
        m60_tangle = (max(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1]) - min(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1])) / c_60m.rolling(20).mean().iloc[-1]
        d_tangle = (max(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1]) - min(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1])) / c_daily.rolling(20).mean().iloc[-1]
        w_tangle = (max(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1]) - min(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1])) / c_weekly.rolling(20).mean().iloc[-1]
        
        if m60_tangle < 0.025 and d_tangle < 0.03 and w_tangle < 0.035 and c_daily.iloc[-1] > c_daily.rolling(20).mean().iloc[-1]:
            return True
    except Exception:
        pass
    return False

def check_extreme_drop_volume_up(df_daily):
    """ 策略四：短線極限超賣 × 爆量紅K (保持原本設定) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        
        c_daily = extract_column_series(df_daily, 'Close')
        o_daily = extract_column_series(df_daily, 'Open')
        v_daily = extract_column_series(df_daily, 'Volume')
        
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        close_today = c_daily.iloc[-1]
        open_today = o_daily.iloc[-1]
        volume_today = v_daily.iloc[-1]
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        
        if rsi6 < 20 and close_today > open_today and volume_today > v_ma5:
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
    
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if bot_token.lower().startswith("bot"): bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"📢 TG 發送反饋: 狀態碼 {res.status_code}")
    except Exception as e: 
        print(f"❌ Telegram 發送異常: {e}")

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股盤後 4 大策略綜合篩選報告（策略一特別融合多週期 KD 低檔金叉）】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    
    strat1_matches, strat2_matches, strat3_matches, strat4_matches = [], [], [], []

    print(f"⏳ 正在進行台股技術面安全分批掃描 (共 {len(tech_scan_pool)} 檔)...")
    for idx, ticker in enumerate(tech_scan_pool, 1):
        if idx % 15 == 0: 
            time.sleep(random.uniform(2.0, 3.5))
            
        try:
            # 1. 統一先下載該股票所有週期數據，減少請求次數並提升效率
            df_daily = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            # 2. 優先進行量能防線檢查 (20日均量 >= 500張)，不符就直接跳過
            if not check_volume_filter(df_daily, min_volume=500): continue
            
            df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False, auto_adjust=True)
            df_weekly = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
            
            if df_60m.empty or df_daily.empty or df_weekly.empty: continue

            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

            # 3. 獨立分流檢測各策略
            if check_strat1_resonance(df_60m, df_daily, df_weekly):
                strat1_matches.append(stock_label)
            if check_oversold_rebound(df_daily):
                strat2_matches.append(stock_label)
            if check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
                strat3_matches.append(stock_label)
            if check_extreme_drop_volume_up(df_daily):
                strat4_matches.append(stock_label)

        except Exception:
            pass

    # 📝 建立獨立美化訊息
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 500張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】原版多週期三頻共振 (MACD + KD 低檔金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "📉 <b>【策略二】季線跌深負乖離 × KD金叉 (中線反彈)</b>\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "💎 <b>【策略三】時/日/週 全週期同步糾結 (變盤極品)</b>\n"
    tw_msg += "↳ " + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "🔥 <b>【策略四】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += "↳ " + (", ".join(strat4_matches) if strat4_matches else "今日無符合標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 台股多策略報告發送完畢！")
