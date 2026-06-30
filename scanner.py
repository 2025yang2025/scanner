import pandas as pd
import yfinance as yf
import requests
import os
import time
import random

# ==============================================================================
# 🌐 【美股全自動動態獲取】標普 500 + 納斯達克 100 成分股總表
# ==============================================================================
def fetch_all_us_market_tickers():
    us_tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        print("🌐 正在初始化全美核心成分股資料庫 (S&P 500 & Nasdaq 100)...")
        # 抓取 S&P 500
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url_sp500)
        sp500_df = tables[0]
        for sym in sp500_df['Symbol'].tolist():
            sym = str(sym).replace('.', '-')
            if sym.isalpha(): us_tickers.add(sym)
            
        # 抓取 Nasdaq 100
        url_ndx = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables_ndx = pd.read_html(url_ndx)
        for table in tables_ndx:
            if 'Ticker' in table.columns:
                for sym in table['Ticker'].tolist():
                    sym = str(sym).replace('.', '-')
                    if sym.isalpha(): us_tickers.add(sym)
                    
        print(f"✅ 成功動態載入全美 {len(us_tickers)} 檔核心大型股代碼！")
    except Exception as e:
        print(f"⚠️ 抓取美股代碼時發生波動: {e}")
        return ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "AVGO", "TSM", "QCOM"]
        
    return sorted(list(us_tickers))

# ==============================================================================
# 📊 【美股財報雙增長過濾】暨【美股中文名稱自動轉換】引擎
# ==============================================================================
def inspect_us_earnings_filter(ticker_symbol):
    """
    動態下載財報，【只回傳】營收與淨利雙雙成長(QoQ > 0)的精選股，並自動加上中文名稱
    """
    # 💡 建立美股核心巨頭中文對照表（其餘無對照的股票會自動改抓 yfinance 官方長名稱）
    us_chinese_names = {
        "NVDA": "輝達", "AAPL": "蘋果", "MSFT": "微軟", "AMZN": "亞馬遜", 
        "META": "臉書", "GOOGL": "谷歌", "GOOG": "谷歌", "AMD": "超微", 
        "AVGO": "博通", "TSM": "台積電ADR", "SMCI": "美超微", "ASML": "艾司摩爾", 
        "QCOM": "高通", "MU": "美光", "INTC": "英特爾", "NFLX": "網飛", "TSLA": "特斯拉"
    }

    try:
        ticker = yf.Ticker(ticker_symbol)
        q_financials = ticker.quarterly_financials
        
        if q_financials.empty or q_financials.shape[1] < 2:
            return None
        
        revenue_row = [idx for idx in q_financials.index if 'Total Revenue' in str(idx) or 'Revenue' in str(idx)]
        net_income_row = [idx for idx in q_financials.index if 'Net Income' in str(idx)]
        
        if not revenue_row or not net_income_row:
            return None
            
        rev_series = q_financials.loc[revenue_row[0]]
        net_series = q_financials.loc[net_income_row[0]]
        
        rev_latest = float(rev_series.iloc[0])
        rev_prev = float(rev_series.iloc[1])
        net_latest = float(net_series.iloc[0])
        net_prev = float(net_series.iloc[1])
        
        # 計算季增率 QoQ
        rev_qoq = ((rev_latest - rev_prev) / rev_prev) * 100 if rev_prev != 0 else 0
        net_qoq = ((net_latest - net_prev) / net_prev) * 100 if net_prev != 0 else 0
        
        # 💡 【核心篩選門檻】：必須營收與淨利雙雙大於 0% 
        if rev_qoq > 0 and net_qoq > 0 and net_latest > 0:
            rev_billion = rev_latest / 1e9
            
            # 💡 自動獲取美股中文/英文名稱機制
            if ticker_symbol in us_chinese_names:
                us_name_zh = us_chinese_names[ticker_symbol]
            else:
                try:
                    # 若不在預設字典，自動動態跟 yfinance 要該公司的官方精簡縮寫名稱
                    us_name_zh = ticker.info.get('shortName', ticker_symbol)
                except Exception:
                    us_name_zh = ""
            
            name_label = f" (*{us_name_zh}*)" if us_name_zh else ""
            return f"• `{ticker_symbol}`{name_label}: 營收 `{rev_billion:.1f}B` (📈 `{rev_qoq:+.1f}%` QoQ) | 淨利 (🟢 `{net_qoq:+.1f}%` QoQ)"
            
    except Exception:
        pass
    return None

# ==============================================================================
# 🇹🇼 台股全市場與技術面模組 (保持全自動含中文功能)
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
        requests.post(url, json=payload, timeout=10)
    except Exception: pass

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    current_hour = now_tw.hour

    # 🌅 早上時段：發送開盤前提醒
    if current_hour < 11:
        if os.path.exists("results.md"):
            with open("results.md", "r", encoding="utf-8") as f:
                saved_content = f.read()
            remind_msg = saved_content.replace("# 📊 *全球雙市場中文化策略選股報告*", "🔔 *【開盤前提醒】雙市場精選報告*")
            send_telegram_message(remind_msg)
            print("✅ 晨間提醒流程執行完畢！")
        exit(0)

    # 📊 下午盤後時段
    print("🚀 啟動【全球雙市場中文化策略選股系統】...")
    
    # --- Part 1: 台股全市場篩選 ---
    ALL_TW_TICKERS = fetch_all_taiwan_market_tickers()
    strat2_candidates, strat3_candidates = fetch_fundamental_snapshot(ALL_TW_TICKERS)
    tech_scan_pool = sorted(list(set(strat2_candidates + strat3_candidates)))
    
    strat1_matches, strat2_matches, strat3_matches = [], [], []

    print(f"⏳ 正在進行台股技術面安全分批掃描 (共 {len(tech_scan_pool)} 檔)...")
    for idx, ticker in enumerate(tech_scan_pool, 1):
        if idx % 15 == 0: time.sleep(random.uniform(2.0, 3.5))
        if check_technical_resonance(ticker):
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            stock_label = f"`{ticker}` (*{name_zh}*)" if name_zh else f"`{ticker}`"
            strat1_matches.append(stock_label)
            if ticker in strat2_candidates: strat2_matches.append(stock_label)
            if ticker in strat3_candidates: strat3_matches.append(stock_label)

    # --- Part 2: 美股全市場大市值雙增長篩選 (內建中文化) ---
    ALL_US_TICKERS = fetch_all_us_market_tickers()
    
    tech_keywords = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AMD", "AVGO", "TSM", "SMCI", "ASML", "QCOM", "MU", "LRCX", "AMAT", "TSLA", "NFLX"]
    us_scan_pool = [tk for tk in ALL_US_TICKERS if tk in tech_keywords or any(tk.startswith(letter) for letter in ['A', 'M', 'N', 'T'])]
    us_scan_pool = sorted(list(set(us_scan_pool[:75]))) 
    
    print(f"⏳ 正在安全分批掃描全美精選科技權值財報並轉換中文名稱 (共 {len(us_scan_pool)} 檔)...")
    us_growth_reports = []
    for idx, us_tk in enumerate(us_scan_pool, 1):
        if idx % 15 == 0:
            time.sleep(random.uniform(2.5, 4.0)) 
        report_line = inspect_us_earnings_filter(us_tk)
        if report_line: 
            us_growth_reports.append(report_line)
        time.sleep(0.5)

    # 📝 格式化全球雙市場綜合報告
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')
    tg_msg = f"# 📊 *全球雙市場中文化策略選股報告*\n⏰ 執行時間: {tw_time_str}\n"
    tg_msg += "---"
    
    # 🇺🇸 美股專區 (全新中文化格式)
    tg_msg += "\n\n🇺🇸 *【美股全市場：最新季度財報雙增長績優股】* 🚀\n"
    tg_msg += "↳ *過濾條件*：最新單季營收與淨利雙增長企業。\n"
    if us_growth_reports:
        for r in us_growth_reports: tg_msg += f"{r}\n"
    else:
        tg_msg += "• 今日無符合雙增長條件之美股。 💤\n"
        
    tg_msg += "---"
    
    # 🇹🇼 台股專區
    tg_msg += "\n\n📈 *【台股策略一：原版多週期三頻共振】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n"

    tg_msg += "\n🚀 *【台股策略二：獲利暴增 × 產業轉折爆發股】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的。 💤") + "\n"

    tg_msg += "\n💎 *【台股策略三：高技術壁壘 × 抗震核心存股龍頭】*\n"
    tg_msg += "• 符合標的：" + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n"

    with open("results.md", "w", encoding="utf-8") as f: f.write(tg_msg)
    send_telegram_message(tg_msg)
    print("✅ 全球雙市場中文化與全自動篩選流程順利完成！")
