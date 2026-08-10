import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇹🇼 台股全市場技術面與基本面模組 (解除產業限制，真正全市場納入)
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}
FUNDAMENTAL_DATA = {}  # 存放 P/E 和 P/B 資料

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼（不限產業），並同步撈取證交所盤後估值資料 """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
    # 1. 撈取全市場基本交易資料
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
    except Exception as e:
        print(f"⚠️ 撈取全市場名單異常: {e}")

    # 2. 撈取證交所官方個股本益比、股價淨值比 (每日盤後更新)
    try:
        url_valuation = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        res_val = requests.get(url_valuation, headers=headers, timeout=10)
        if res_val.status_code == 200:
            for item in res_val.json():
                code = item.get("Code", "").strip()
                ticker_id = f"{code}.TW"
                
                try:
                    pe = float(item.get("PEratio", 0)) if item.get("PEratio") else 0.0
                except:
                    pe = 0.0
                try:
                    pb = float(item.get("PBRatio", 0)) if item.get("PBRatio") else 0.0
                except:
                    pb = 0.0
                
                FUNDAMENTAL_DATA[ticker_id] = {"PE": pe, "PB": pb}
    except Exception as e:
        print(f"⚠️ 撈取證交所估值資料異常: {e}")

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

def calculate_kd(df_single, n=9, m1=3, m2=3):
    low_min = df_single['Low'].astype(float).rolling(window=n).min()
    high_max = df_single['High'].astype(float).rolling(window=n).max()
    close = df_single['Close'].astype(float)
    
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    rsv = rsv.fillna(50)
    
    k_list, d_list = [50.0], [50.0]
    for i in range(1, len(rsv)):
        current_k = (k_list[-1] * (m1 - 1) + rsv.iloc[i]) / m1
        current_d = (d_list[-1] * (m2 - 1) + current_k) / m2
        k_list.append(current_k)
        d_list.append(current_d)
        
    return pd.Series(k_list, index=df_single.index), pd.Series(d_list, index=df_single.index)

def calculate_rsi(close_series, period=6):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ==============================================================================
# 🎯 核心策略檢測邏輯
# ==============================================================================

def check_macd_up_and_kd_gold(df_single):
    """ 通用模組：MACD 往 0 軸向上 + KD 黃金交叉 """
    try:
        if df_single.empty or len(df_single) < 26: return False
        c = df_single['Close'].squeeze().astype(float)
        
        # 1. MACD 計算
        macd_line, signal_line, hist = calculate_macd(c)
        if len(macd_line) < 2: return False
        
        # MACD 條件：快線 (DIF) 持續向上，且向 0 軸靠近或在 0 軸之上
        macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (
            macd_line.iloc[-1] >= 0 or (hist.iloc[-1] > hist.iloc[-2])
        )
        
        # 2. KD 計算與金叉判斷
        k_ser, d_ser = calculate_kd(df_single)
        if len(k_ser) < 2: return False
        
        kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
        
        return macd_up and kd_gold
    except Exception:
        return False

def check_strat1_resonance(df_30m, df_60m):
    """ 策略一：30分K 與 60分K 同步 MACD往0軸向上 × KD黃金交叉 """
    try:
        return check_macd_up_and_kd_gold(df_30m) and check_macd_up_and_kd_gold(df_60m)
    except Exception:
        pass
    return False

def check_strat2_resonance(df_daily, df_weekly):
    """ 策略二（新）：日K 與 週K 同步 MACD往0軸向上 × KD黃金交叉 """
    try:
        return check_macd_up_and_kd_gold(df_daily) and check_macd_up_and_kd_gold(df_weekly)
    except Exception:
        pass
    return False

def check_volume_breakout(df_daily):
    """ 策略三（新，原策略七）：關鍵均線多頭突破 × 量能倍增 (帶量突破) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        close_today = c_daily.iloc[-1]
        close_yesterday = c_daily.iloc[-2]
        ma20_today = ma20.iloc[-1]
        ma20_yesterday = ma20.iloc[-2]
        
        price_break_cond = (close_today > ma20_today) and (close_yesterday <= ma20_yesterday or (close_today - close_yesterday) / close_yesterday > 0.02)
        if not price_break_cond: return False
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        volume_cond = volume_today > (v_ma5 * 1.5)
        if not volume_cond: return False
        
        k_series, d_series = calculate_kd(df_daily)
        k_today = k_series.iloc[-1]
        d_today = d_series.iloc[-1]
        kd_cond = (k_today > d_today) and (k_today < 75)
        
        if kd_cond:
            volume_ratio = volume_today / v_ma5 if v_ma5 > 0 else 1.0
            return True, volume_ratio
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

def check_bollinger_squeeze_fast(df_daily):
    """ 策略五：布林軌道極致壓縮 """
    try:
        c_daily = df_daily['Close'].squeeze().astype(float)
        ma20 = c_daily.rolling(window=20).mean()
        std20 = c_daily.rolling(window=20).std()
        
        bandwidth = (4 * std20) / ma20
        current_bw = bandwidth.iloc[-1]
        
        current_close = c_daily.iloc[-1]
        current_ma20 = ma20.iloc[-1]
        current_up_band = current_ma20 + (2 * std20.iloc[-1])
        
        cond_squeeze = current_bw <= 0.06
        cond_above_ma = current_close >= current_ma20
        cond_near_upper = ((current_up_band - current_close) / current_close) <= 0.02
        
        if cond_squeeze and cond_above_ma and cond_near_upper:
            return True, round(current_bw * 100, 2)
            
    except Exception:
        pass
        
    return False, 0

def check_strat6_undervalued(ticker):
    """ 策略六：基本面價值型低估股 """
    data = FUNDAMENTAL_DATA.get(ticker)
    if not data: return False
    
    pe = data.get("PE", 0)
    pb = data.get("PB", 0)
    
    if 0 < pe <= 12.0 and 0 < pb <= 1.0:
        return True, pe, pb
    return False

def check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    """ 策略七（新，原策略三）：60分K/日K/週K同步均線糾結 """
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

    print("🚀 啟動【台股盤後 7 大策略全市場篩選報告】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    
    if not tech_scan_pool:
        print("❌ 未能取得任何股票代碼，程式結束。")
        exit()

    # ─── 【量能篩選門檻：1000 張】 ───
    print(f"⏳ 步驟 1: 批次下載全市場日K資料進行量能過濾 (門檻：20日均量 >= 1000張，共 {len(tech_scan_pool)} 檔)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    qualified_tickers = []
    for ticker in tech_scan_pool:
        try:
            if len(tech_scan_pool) == 1:
                v_daily = full_df_daily['Volume'].squeeze()
            else:
                if ticker not in full_df_daily.columns.levels[1]: continue
                v_daily = full_df_daily.xs(ticker, axis=1, level=1)['Volume'].squeeze()
                
            if len(v_daily) >= 20:
                v_ma20_sheets = v_daily.rolling(window=20).mean().iloc[-1] / 1000
                if v_ma20_sheets >= 1000:
                    qualified_tickers.append(ticker)
        except Exception:
            continue

    print(f"🎯 通過量能防線股票共 {len(qualified_tickers)} 檔。")
    
    strat1_matches, strat2_matches, strat3_matches, strat4_matches, strat5_matches, strat6_matches, strat7_matches = [], [], [], [], [], [], []

    if qualified_tickers:
        print("⏳ 步驟 2: 批次下載精選股票的 30分K、60分K 與 週K 資料...")
        full_df_30m = yf.download(qualified_tickers, period="1mo", interval="30m", progress=False, auto_adjust=True)
        full_df_60m = yf.download(qualified_tickers, period="1mo", interval="60m", progress=False, auto_adjust=True)
        full_df_weekly = yf.download(qualified_tickers, period="2y", interval="1wk", progress=False, auto_adjust=True)

        print("⏳ 步驟 3: 記憶體內高速多維策略流檢測中...")
        for ticker in qualified_tickers:
            try:
                if len(qualified_tickers) == 1:
                    df_d = full_df_daily.copy()
                    df_m30 = full_df_30m.copy()
                    df_m60 = full_df_60m.copy()
                    df_w = full_df_weekly.copy()
                else:
                    if ticker not in full_df_daily.columns.levels[1]: continue
                    if ticker not in full_df_30m.columns.levels[1]: continue
                    if ticker not in full_df_60m.columns.levels[1]: continue
                    if ticker not in full_df_weekly.columns.levels[1]: continue

                    df_d = full_df_daily.xs(ticker, axis=1, level=1)
                    df_m30 = full_df_30m.xs(ticker, axis=1, level=1)
                    df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                    df_w = full_df_weekly.xs(ticker, axis=1, level=1)

                if df_d.empty or df_m30.empty or df_m60.empty or df_w.empty: 
                    continue

                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 策略一：30分K + 60分K 共振
                if check_strat1_resonance(df_m30, df_m60):
                    strat1_matches.append(stock_label)
                    
                # 策略二：日K + 週K 共振
                if check_strat2_resonance(df_d, df_w):
                    strat2_matches.append(stock_label)

                # 策略三：帶量突破 (原策略七)
                vol_breakout_check = check_volume_breakout(df_d)
                if vol_breakout_check:
                    _, vol_ratio = vol_breakout_check
                    strat3_matches.append(f"{stock_label}[量比:{vol_ratio:.1f}倍]")

                # 策略四：極限超賣爆量
                if check_extreme_drop_volume_up(df_d):
                    strat4_matches.append(stock_label)
                
                # 策略五：布林壓縮
                boll_check, bw_val = check_bollinger_squeeze_fast(df_d)
                if boll_check:
                    strat5_matches.append(f"{stock_label}[帶寬:{bw_val:.1f}%]")
                
                # 策略六：價值低估
                val_check = check_strat6_undervalued(ticker)
                if val_check:
                    _, cur_pe, cur_pb = val_check
                    strat6_matches.append(f"{stock_label}[PE:{cur_pe:.1f}, PB:{cur_pb:.2f}]")

                # 策略七：全週期糾結 (原策略三)
                if check_multi_timeframe_tangling(df_m60, df_d, df_w):
                    strat7_matches.append(stock_label)

            except KeyError:
                continue
            except Exception as e:
                print(f"⚠️ 處理個股 {ticker} 時發生未預期錯誤: {e}")
                continue

    # 📝 建立 7 大策略綜合美化報告訊息
    tw_msg = f"🇹🇼 <b>【台股多策略選股報告】</b>\n⚠️ <i>已過濾 20日均量 &lt; 1000張之殭屍股</i>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】30分K & 60分K 共振 (MACD 往0軸向上 + KD金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "📈 <b>【策略二】日K & 週K 共振 (MACD 往0軸向上 + KD金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "⚡ <b>【策略三】關鍵均線多頭突破 × 量能倍增 (帶量突破)</b>\n"
    tw_msg += "↳ " + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "🔥 <b>【策略四】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += "↳ " + (", ".join(strat4_matches) if strat4_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "🚀 <b>【策略五】布林軌道極致壓縮 (帶寬 ≤ 6% × 貼近上軌)</b>\n"
    tw_msg += "↳ " + (", ".join(strat5_matches) if strat5_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "💰 <b>【策略六】價值型低估股 (本益比 ≤ 12 × 股價淨值比 ≤ 1.0)</b>\n"
    tw_msg += "↳ " + (", ".join(strat6_matches) if strat6_matches else "今日無符合標的。 💤") + "\n\n"

    tw_msg += "💎 <b>【策略七】時/日/週 全週期同步糾結 (不限排列)</b>\n"
    tw_msg += "↳ " + (", ".join(strat7_matches) if strat7_matches else "今日無符合標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 台股多策略基本面綜合報告發送完畢！")
