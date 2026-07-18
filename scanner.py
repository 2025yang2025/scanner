import os
import sys
import datetime
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# ⚙️ 系統基本設定
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_tg(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ 錯誤：未設定 Telegram 憑證，僅在終端機輸出。")
        print(text)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        return True
    except Exception as e:
        print(f"❌ Telegram 發送失敗: {e}")
        return False

# ==========================================
# 📈 技術指標計算工具
# ==========================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_kd(df, n=9, m1=3, m2=3):
    # 確保有足夠數據計算 KD
    if len(df) < n:
        return pd.Series([50]*len(df)), pd.Series([50]*len(df))
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    rsv = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    rsv = rsv.fillna(50)
    
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    return k, d

def calculate_rsi(close_series, period=6):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==========================================
# 🎯 核心策略檢測邏輯
# ==========================================
def check_strat1_resonance(df_60m, df_daily, df_weekly):
    """ 策略一：原版多週期三頻共振 (MACD) + KD低檔金叉 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
        if c_60m.empty or c_daily.empty or c_weekly.empty: return False

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
    """ 策略二：季線跌深負乖離 × KD金叉 (此處沿用原代碼註解的策略二命名) """
    try:
        if df_daily.empty or len(df_daily) < 60: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
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
    """ 策略三：60分K/日K/週K同步均線糾結 """
    try:
        c_60m = df_60m['Close'].squeeze().astype(float)
        c_daily = df_daily['Close'].squeeze().astype(float)
        c_weekly = df_weekly['Close'].squeeze().astype(float)
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
    """ 策略四：短線極限超賣 × 爆量紅K """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
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

# ==========================================
# ⚡ 全港股極速粗篩（流動性過濾與中文對照）
# ==========================================
def fetch_hk_shortlist_auto():
    print("🌐 正在抓取港股流動性數據並建立中文名稱字典...")
    shortlist = []
    name_dict = {}
    
    for page in range(1, 7):
        url = f"https://vip.stock.finance.sina.com.cn/hq/api/jsonp.php/IO.XSRV2.CallbackList['hk']/HK_Service.getMainMethodPageList?page={page}&num=80&sort=amount&asc=0"
        try:
            response = requests.get(url, timeout=10)
            text = response.text
            if "bracket" in text or "CallbackList" in text:
                left = text.find("[")
                right = text.rfind("]") + 1
                text = text[left:right]
            
            data = pd.read_json(text)
            if data.empty:
                break
                
            for _, row in data.iterrows():
                raw_code = str(row['symbol'])
                pure_code = raw_code[-4:] if len(raw_code) == 5 else raw_code
                ticker = f"{int(pure_code):04d}.HK"
                
                name_dict[pure_code] = row['name']
                trade = float(row['trade'])      
                turnover = float(row['amount'])  
                
                # 基礎門檻粗篩
                if trade >= 1.0 and turnover >= 8000000:
                    shortlist.append(ticker)
        except:
            continue
            
    shortlist = list(set(shortlist))
    
    if not shortlist:
        print("⚠️ 偵測到當前可能為休市期間，使用核心活躍股作為基本池...")
        backup_tickers = []
        core_list = [
            (1, "長江和記"), (5, "匯豐控股"), (388, "香港交易所"), (700, "騰訊控股"), 
            (941, "中國移動"), (1211, "比亞迪股份"), (1810, "小米集團-W"), (2015, "理想汽車-W"), 
            (2318, "中國平安"), (3690, "美團-W"), (9618, "京東集團-SW"), (9988, "阿里巴巴-W")
        ]
        for code, name in core_list:
            pure_code = f"{code:04d}"
            ticker = f"{pure_code}.HK"
            backup_tickers.append(ticker)
            if pure_code not in name_dict:
                name_dict[pure_code] = name
        return backup_tickers, name_dict
        
    return shortlist, name_dict

# ==========================================
# 🚀 深度多週期技術面篩選
# ==========================================
def scan_all_hong_kong_market_fast():
    start_time = time.time()
    
    shortlist, name_dict = fetch_hk_shortlist_auto()
    print(f"\n🚀 【深度分析階段】正在下載與分析 {len(shortlist)} 檔標的之 60分K/日K/週K 數據...")
    
    # 儲存命中各核心策略的結果
    hit_strat1 = []
    hit_strat2 = []
    hit_strat3 = []
    hit_strat4 = []
    
    # 為提高穩定性與速度，此處採單檔分流下載與技術型態檢測
    for ticker in shortlist:
        try:
            pure_code = ticker.split('.')[0]
            stock_name = name_dict.get(pure_code, "未知名稱")
            
            # 1. 抓取多週期數據
            # 60分K 需要近期的 bar（取 1個月內足夠算 20 MA、KD、MACD）
            df_60m = yf.download(ticker, period="1mo", interval="60m", progress=False, ignore_tz=True)
            # 日K 取 6個月（滿足策略二 60日均線）
            df_daily = yf.download(ticker, period="6mo", interval="1d", progress=False, ignore_tz=True)
            # 週K 取 1年
            df_weekly = yf.download(ticker, period="1y", interval="1wk", progress=False, ignore_tz=True)
            
            if df_daily.empty or len(df_daily) < 2:
                continue
                
            # 基本即時數據計算（做發送呈現用）
            prev_close = df_daily["Close"].squeeze().astype(float).iloc[-2]
            today_close = df_daily["Close"].squeeze().astype(float).iloc[-1]
            today_high = df_daily["High"].squeeze().astype(float).iloc[-1]
            change_percent = ((today_close - prev_close) / prev_close) * 100
            today_volume = df_daily["Volume"].squeeze().astype(float).iloc[-1]
            turnover = today_volume * today_close
            
            stock_info = {
                "id": pure_code,
                "name": stock_name,
                "close": today_close,
                "change": change_percent,
                "turnover": turnover
            }

            # 2. 核心策略多重條件檢測
            # 策略 A：原「爆量突破收最高」
            is_breakout_highest = (change_percent >= 5.0) and ((today_high - today_close) <= (today_close * 0.005))
            if is_breakout_highest:
                hit_strat1.append(stock_info)
                
            # 策略一：原版多週期三頻共振
            if check_strat1_resonance(df_60m, df_daily, df_weekly):
                hit_strat2.append(stock_info)
                
            # 策略二：季線跌深負乖離 × KD金叉
            if check_oversold_rebound(df_daily):
                hit_strat3.append(stock_info)
                
            # 策略三：多週期同步均線糾結
            if check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
                hit_strat4.append(stock_info)
                
            # 註：此處可依需求增加對 check_extreme_drop_volume_up (策略四) 的檢測分流
                
        except Exception as e:
            # 個股計算異常則跳過，不中斷主流程
            continue

    # ==========================================
    # 📊 整合單一 Telegram 訊息輸出
    # ==========================================
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    full_report_msg = f"📋 *【港股多週期核心策略綜合報告】* ({today_str})\n"
    full_report_msg += "========================\n\n"

    # --- 策略 A 顯示 ---
    full_report_msg += "🚀 *一、強勢動能：突破收最高*\n"
    if hit_strat1:
        # 按漲幅排序取前 5
        hit_strat1_sorted = sorted(hit_strat1, key=lambda x: x['change'], reverse=True)[:5]
        for s in hit_strat1_sorted:
            full_report_msg += f"📌 *{s['id']} {s['name']}*\n💰 價格：`{s['close']:.2f} HKD` (`+{s['change']:.2f}%`)\n📊 成交額：`{s['turnover']/1000000:.1f}M HKD`\n"
    else:
        full_report_msg += "👉 _今日暫無符合標的。_\n"
    full_report_msg += "\n------------------------\n\n"

    # --- 策略一顯示 ---
    full_report_msg += "🎯 *二、三頻共振：MACD多週期 × KD低金*\n"
    if hit_strat2:
        for s in hit_strat2[:5]:
            full_report_msg += f"🔥 *{s['id']} {s['name']}*\n💰 價格：`{s['close']:.2f} HKD` (`{s['change']:.2f}%`)\n"
    else:
        full_report_msg += "👉 _今日暫無符合標的。_\n"
    full_report_msg += "\n------------------------\n\n"

    # --- 策略二顯示 ---
    full_report_msg += "📉 *三、超跌反彈：季線負乖離 × KD金叉*\n"
    if hit_strat3:
        for s in hit_strat3[:5]:
            full_report_msg += f"🩹 *{s['id']} {s['name']}*\n💰 價格：`{s['close']:.2f} HKD` (`{s['change']:.2f}%`)\n"
    else:
        full_report_msg += "👉 _今日暫無符合標的。_\n"
    full_report_msg += "\n------------------------\n\n"

    # --- 策略三顯示 ---
    full_report_msg += "🌀 *四、蓄勢待發：多週期均線同步糾結*\n"
    if hit_strat4:
        for s in hit_strat4[:5]:
            full_report_msg += f"📦 *{s['id']} {s['name']}*\n💰 價格：`{s['close']:.2f} HKD` (`{s['change']:.2f}%`)\n"
    else:
        full_report_msg += "👉 _今日暫無符合標的。_\n"
    
    full_report_msg += "\n========================"

    # 一鍵發送整份整合報告
    send_tg(full_report_msg)
    print(f"🎉 核心策略整合報告發送結束！總耗時: {time.time() - start_time:.1f} 秒")

if __name__ == "__main__":
    scan_all_hong_kong_market_fast()
