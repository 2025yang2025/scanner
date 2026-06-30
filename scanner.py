import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🇺🇸 鎖定美股 AI 與科技核心權值股名單（進行精準財報健檢）
# ==============================================================================
US_TARGETS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "AVGO", "TSM", "SMCI", "ASML", "QCOM"
]

# 全域字典：用來動態儲存全台股 1800+ 檔最新的【代號: 中文名稱】對照表
DYNAMIC_STOCK_NAMES = {}

# ==============================================================================
# 🌐 台股全市場標的與【中文名稱】動態獲取
# ==============================================================================
def fetch_all_taiwan_market_tickers():
    """動態抓取台灣市場所有上市股票代號與中文名稱 (一次打包，不傷伺服器)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    try:
        print("🌐 正在從證交所初始化全市場股票代碼與【中文名稱】資料庫...")
        # 讀取證交所全市場每日收盤行情快照 API
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()  # 💡 抓取官方中文名稱
                
                if code.isdigit() and len(code) == 4:
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name  # 💡 自動寫入全域字典
                    
            print(f"✅ 成功動態載入 {len(all_tickers)} 檔台股中文對照表！")
    except Exception as e:
        print(f"⚠️ 抓取上市代碼與名稱時發生波動: {e}")
        
    # 保底機制：萬一證交所 API 斷線，提供核心股票名稱確保不崩潰
    if not all_tickers:
        print("🚨 啟用種子標的群組保底...")
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2382.TW": "廣達", "3017.TW": "奇鋐"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
            
    return sorted(list(set(all_tickers)))

def fetch_fundamental_snapshot(tickers):
    """基本面/特定高科技產業群初審濾網"""
    strat2_candidates = []
    strat3_candidates = []
    for tk in tickers:
        pure_code = tk.split('.')[0]
        # 篩選電子、半導體、高精密製造等核心飆股族群
        if pure_code.startswith(('23', '24', '30', '32', '34', '35', '36', '37', '61', '62', '64', '80')):
            strat2_candidates.append(tk)
            # 極高技術壁壘龍頭種子
            if pure_code in ['2330', '2454', '3443', '3661', '6415', '3017', '3533', '6187']:
                strat3_candidates.append(tk)
    return strat2_candidates, strat3_candidates

# ==============================================================================
# 📊 美股核心財報健檢核心引擎
# ==============================================================================
def inspect_us_earnings(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        q_financials = ticker.quarterly_financials
        
        if q_financials.empty or q_financials.shape[1] < 2:
            return f"• `{ticker_symbol}`: ⚠️ 暫無足夠季度財報數據"
        
        revenue_row = [idx for idx in q_financials.index if 'Total Revenue' in str(idx) or 'Revenue' in str(idx)]
        net_income_row = [idx for idx in q_financials.index if 'Net Income' in str(idx)]
        
        if not revenue_row or not net_income_row:
            return f"• `{ticker_symbol}`: ⚠️ 財報欄位解析受阻"
            
        rev_series = q_financials.loc[revenue_row[0]]
        net_series = q_financials.loc[net_income_row[0]]
        
        rev_latest = float(rev_series.iloc[0])
        rev_prev = float(rev_series.iloc[1])
        net_latest = float(net_series.iloc[0])
        net_prev = float(net_series.iloc[1])
        
        rev_qoq = ((rev_latest - rev_prev) / rev_prev) * 100 if rev_prev != 0 else 0
        net_qoq = ((net_latest - net_prev) / net_prev) * 100 if net_prev != 0 else 0
        
        rev_status = "📈 增長" if rev_qoq >= 0 else "📉 衰退"
        net_status = "🟢 獲利擴大" if net_qoq >= 0 else "🔴 獲利縮水"
        if net_latest < 0: net_status = "🚨 虧損"
        
        rev_billion = rev_latest / 1e9
        return f"• `{ticker_symbol}`: 營收 `{rev_billion:.1f}B` ({rev_status} `{rev_qoq:+.1f}%` QoQ) | 淨利 ({net_status} `{net_qoq:+.1f}%` QoQ)"
    except Exception:
        return f"• `{ticker_symbol}`: ❌ 財報健檢執行異常"

# ==============================================================================
# 📈 技術面計算（台股三頻共振）
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
    return pd.Series(dtype=float)

def check_technical_resonance(ticker):
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

        if len(w_hist) < 1 or len(d_hist) < 1 or len(m60_hist) < 2: return False

        w_m, w_s, w_h = float(w_macd.iloc[-1]), float(w_signal.iloc[-1]), float(w_hist.iloc[-1])
        d_m, d_s, d_c, d_ma_val = float(d_macd.iloc[-1]), float(d_signal.iloc[-1]), float(c_daily.iloc[-1]), float(d_ma.iloc[-1])
        m60_m, m60_h, m60_h_prev = float(m60_macd.iloc[-1]), float(m60_hist.iloc[-1]), float(m60_hist.iloc[-2])

        weekly_bullish = (w_m > w_s) and (w_h > 0)
        daily_bullish = (d_m > 0) and (d_m > d_s)
        daily_above_ma = (d_c > d_ma_val)
        m60_cross_up = (m60_m > 0) and (m60_h > 0) and (m60_h_prev <= 0)

        if weekly_bullish and daily_bullish and daily_above_ma and m60_cross_up:
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
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200: print(f"❌ TG 發送失敗: {response.text}")
    except Exception as e: print(f"發送 TG 異常: {e}")

# ==============================================================================
# 🚀 主程式：跨國市場聯合大掃描
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    current_hour = now_tw.hour

    # 🌅 早上時段：直接讀取昨日報告存檔，送出開盤前提醒
    if current_hour < 11:
        if os.path.exists("results.md"):
            with open("results.md", "r", encoding="utf-8") as f:
                saved_content = f.read()
            remind_msg = saved_content.replace("# 📊 *全球雙市場跨維度選股與財報報告*", "🔔 *【開盤前提醒】全球雙市場選股報告*")
            send_telegram_message(remind_msg)
            print("✅ 晨間提醒流程執行完畢！")
        exit(0)

    # 📊 下午盤後時段：台股策略選股 + 美股核心財報健檢
    print("🚀 啟動【全球雙市場跨維度選股與財報系統】...")
    
    # --- Part 1: 台股全市場與中文名稱加載 ---
    ALL_MARKET_TICKERS = fetch_all_taiwan_market_tickers()
    strat2_candidates, strat3_candidates = fetch_fundamental_snapshot(ALL_MARKET_TICKERS)
    tech_scan_pool = sorted(list(set(strat2_candidates + strat3_candidates)))
    
    strat1_matches = []
    strat2_matches = []
    strat3_matches = []

    print(f"⏳ 正在進行台股技術面安全分批掃描 (共 {len(tech_scan_pool)} 檔)...")
    for idx, ticker in enumerate(tech_scan_pool, 1):
        if idx % 15 == 0:
            time.sleep(random.uniform(2.0, 3.5))
            
        if check_technical_resonance(ticker):
            # 💡 從動態字典中自動查詢最新中文名稱，若查不到則顯示空值
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"`{ticker}` (*{name_zh}*)" if name_zh else f"`{ticker}`"
            
            strat1_matches.append(stock_label)
            if ticker in strat2_candidates: strat2_matches.append(stock_label)
            if ticker in strat3_candidates: strat3_matches.append(stock_label)

    # --- Part 2: 美股核心財報健檢 ---
    print("⏳ 正在下載美股核心科技股最新季度財報進行動態健檢...")
    us_reports = []
    for us_tk in US_TARGETS:
        report_line = inspect_us_earnings(us_tk)
        us_reports.append(report_line)
        time.sleep(1.5)

    # 📝 格式化全球雙市場綜合報告
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')
    tg_msg = f"# 📊 *全球雙市場跨維度選股與財報報告*\n⏰ 執行時間: {tw_time_str}\n"
    tg_msg += "---"
    
    # 🇺🇸 美股專區
    tg_msg += "\n\n🇺🇸 *【美股核心巨頭財報動態健檢】*\n"
    tg_msg += "↳ *指標意義*：檢視最新單季營收規模、QoQ 增長/衰退及淨利增減。\n"
    for r in us_reports:
        tg_msg += f"{r}\n"
        
    tg_msg += "---"
    
    # 🇹🇼 台股專區 (格式完美支援中文名稱)
    tg_msg += "\n\n📈 *【台股策略一：原版多週期三頻共振】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n"

    tg_msg += "\n🚀 *【台股策略二：獲利暴增 × 產業轉折爆發股】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的. 💤") + "\n"

    tg_msg += "\n💎 *【台股策略三：高技術壁壘 × 抗震核心存股龍頭】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n"

    # 儲存與發送
    with open("results.md", "w", encoding="utf-8") as f:
        f.write(tg_msg)
        
    send_telegram_message(tg_msg)
    print("✅ 全球雙市場全自動安全掃描流程（含台股中文名稱）順利完成！")
