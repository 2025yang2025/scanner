# ============================================================
# Taiwan Stock Scanner Pro v2.1
# MTF Hybrid Version
#
# 策略：
# S1 = 30分K MACD 負值縮小 + KD > 20
# S2 = 60分K MACD 負值縮小 + KD > 20
# S3 = 日K MACD > 0 + KD > 20
# S4 = 週K MACD 突破0軸後維持多方 + KD > 50
# S5 = 月K MACD 突破0軸後維持多方 + KD > 50
# S6 = S3 + S4 + S5 多週期共振
#
# 原策略六「低檔爆量」已刪除
#
# 原始核心：
# 月K → 週K → 日K → 60分K
#
# 最終強勢觸發：
# 60分K MACD 綠柱縮小 → 綠柱轉紅
# ============================================================

import os
import time
import html
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import requests
import yfinance as yf


# ============================================================
# 版本
# ============================================================

VERSION = "Pro v2.1 MTF"

# ============================================================
# TWSE
# ============================================================

TWSE_API = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

# ============================================================
# 基本設定
# ============================================================

INITIAL_CAPITAL = 300000

MIN_AVG_VOLUME_20 = 1_000_000

# 日K → 先篩選流動性及強度
INTRADAY_SCAN_LIMIT = 100

# yfinance 分批
DAILY_CHUNK_SIZE = 150
INTRADAY_CHUNK_SIZE = 50

# ============================================================
# KD
# ============================================================

DAILY_KD_THRESHOLD = 20
WEEKLY_KD_THRESHOLD = 50
MONTHLY_KD_THRESHOLD = 50
INTRADAY_KD_THRESHOLD = 20

# ============================================================
# 多週期 MACD 0軸突破設定
# ============================================================

# 週K：
# 最近幾根K之內曾突破0軸，目前仍 > 0
WEEKLY_ZERO_CROSS_LOOKBACK = 6

# 月K：
# 最近幾個月之內曾突破0軸，目前仍 > 0
MONTHLY_ZERO_CROSS_LOOKBACK = 3

# 60K：
# 綠柱至少連續縮小的概念
M60_TRIGGER_LOOKBACK = 2

# ============================================================
# 股票名稱快取
# ============================================================

DYNAMIC_STOCK_NAMES = {}


# ============================================================
# 基本工具
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        if value == "":
            return default

        return float(value)

    except Exception:
        return default


def get_ticker_code(ticker):
    if ticker is None:
        return ""

    ticker = str(ticker)

    if ticker.endswith(".TW"):
        ticker = ticker[:-3]

    return ticker


def escape_html(text):
    return html.escape(str(text), quote=False)


def get_stock_label(code, name=None):
    code = get_ticker_code(code)

    if name:
        return f"{code} {name}"

    if code in DYNAMIC_STOCK_NAMES:
        return f"{code} {DYNAMIC_STOCK_NAMES[code]}"

    return code


# ============================================================
# TWSE 股票清單
# ============================================================

def fetch_all_taiwan_market_tickers():
    print("📥 取得台股股票清單...")

    try:
        response = requests.get(
            TWSE_API,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        tickers = []

        for item in data:

            code = str(item.get("Code", "")).strip()
            name = str(item.get("Name", "")).strip()

            # 只抓一般4碼股票
            if len(code) != 4:
                continue

            if not code.isdigit():
                continue

            # 排除 ETF / 特殊商品等
            if code.startswith(("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")):
                continue

            DYNAMIC_STOCK_NAMES[code] = name

            tickers.append(f"{code}.TW")

        tickers = sorted(list(set(tickers)))

        print(f"✅ 找到 {len(tickers)} 檔股票")

        return tickers

    except Exception as e:
        print(f"❌ 取得股票清單失敗：{e}")
        return []


# ============================================================
# yfinance
# ============================================================

def safe_download_yf(
    tickers,
    period="1y",
    interval="1d",
    chunk_size=150,
):
    """
    分批下載 Yahoo Finance。
    回傳：
        {
            "2330.TW": dataframe,
            ...
        }
    """

    results = {}

    if not tickers:
        return results

    tickers = list(dict.fromkeys(tickers))

    print(
        f"📊 Yahoo Finance下載 "
        f"period={period}, interval={interval}, "
        f"stocks={len(tickers)}"
    )

    for start in range(0, len(tickers), chunk_size):

        chunk = tickers[start:start + chunk_size]

        print(
            f"   → {start + 1}-{min(start + len(chunk), len(tickers))}"
            f"/{len(tickers)}"
        )

        try:
            data = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )

            if data is None or data.empty:
                continue

            # ------------------------------------------------
            # MultiIndex
            # ------------------------------------------------

            if isinstance(data.columns, pd.MultiIndex):

                level0 = list(data.columns.get_level_values(0))
                level1 = list(data.columns.get_level_values(1))

                ticker_set = set(chunk)

                # 情況1：
                # Ticker / Open / High...
                if any(x in ticker_set for x in level0):

                    for ticker in chunk:

                        try:
                            if ticker in data.columns.get_level_values(0):
                                df = data[ticker].copy()
                            else:
                                continue

                            if not df.empty:
                                results[ticker] = normalize_dataframe(df)

                        except Exception:
                            continue

                # 情況2：
                # Open / High / ... / Ticker
                else:

                    for ticker in chunk:

                        try:
                            if ticker in data.columns.get_level_values(1):

                                df = data.xs(
                                    ticker,
                                    axis=1,
                                    level=1,
                                ).copy()

                                if not df.empty:
                                    results[ticker] = normalize_dataframe(df)

                        except Exception:
                            continue

            else:

                # 單一股票
                if len(chunk) == 1:

                    ticker = chunk[0]

                    df = data.copy()

                    if not df.empty:
                        results[ticker] = normalize_dataframe(df)

        except Exception as e:
            print(f"   ⚠️ Yahoo下載失敗：{e}")

        time.sleep(0.3)

    print(f"✅ 成功取得 {len(results)} 檔")

    return results


def normalize_dataframe(df):

    df = df.copy()

    # 如果欄位還是 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):

        new_columns = []

        for col in df.columns:

            if isinstance(col, tuple):
                found = None

                for x in col:
                    if str(x) in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume",
                    ]:
                        found = str(x)
                        break

                new_columns.append(
                    found if found else str(col[-1])
                )

            else:
                new_columns.append(str(col))

        df.columns = new_columns

    # 統一欄位名稱
    rename_map = {}

    for col in df.columns:

        c = str(col).strip().lower()

        if c == "open":
            rename_map[col] = "Open"

        elif c == "high":
            rename_map[col] = "High"

        elif c == "low":
            rename_map[col] = "Low"

        elif c == "close":
            rename_map[col] = "Close"

        elif c == "adj close":
            rename_map[col] = "Adj Close"

        elif c == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in required:

        if col not in df.columns:
            df[col] = np.nan

    df = df[required].copy()

    df = df.sort_index()

    df = df[~df.index.duplicated(keep="last")]

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    return df


# ============================================================
# MACD
# ============================================================

def calculate_macd(
    close,
    fast=12,
    slow=26,
    signal=9,
):

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    ema_fast = close.ewm(
        span=fast,
        adjust=False,
        min_periods=fast,
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False,
        min_periods=slow,
    ).mean()

    macd = ema_fast - ema_slow

    signal_line = macd.ewm(
        span=signal,
        adjust=False,
        min_periods=signal,
    ).mean()

    hist = macd - signal_line

    return macd, signal_line, hist


# ============================================================
# KD
# ============================================================

def calculate_kd(
    df,
    period=9,
    k_smooth=3,
    d_smooth=3,
):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    lowest_low = low.rolling(
        period,
        min_periods=period,
    ).min()

    highest_high = high.rolling(
        period,
        min_periods=period,
    ).max()

    denominator = highest_high - lowest_low

    rsv = (
        (close - lowest_low)
        / denominator.replace(0, np.nan)
        * 100
    )

    rsv = rsv.fillna(50)

    k = rsv.ewm(
        alpha=1 / k_smooth,
        adjust=False,
    ).mean()

    d = k.ewm(
        alpha=1 / d_smooth,
        adjust=False,
    ).mean()

    return k, d


# ============================================================
# KD 狀態
# ============================================================

def get_kd_state(df):

    if df is None or len(df) < 15:
        return {
            "k": np.nan,
            "d": np.nan,
            "k_prev": np.nan,
            "d_prev": np.nan,
            "golden_cross": False,
        }

    k, d = calculate_kd(df)

    if len(k) < 2:
        return {
            "k": np.nan,
            "d": np.nan,
            "k_prev": np.nan,
            "d_prev": np.nan,
            "golden_cross": False,
        }

    return {
        "k": float(k.iloc[-1]),
        "d": float(d.iloc[-1]),
        "k_prev": float(k.iloc[-2]),
        "d_prev": float(d.iloc[-2]),
        "golden_cross": (
            k.iloc[-2] <= d.iloc[-2]
            and k.iloc[-1] > d.iloc[-1]
        ),
    }


# ============================================================
# MACD 狀態
# ============================================================

def get_macd_state(df):

    if df is None or len(df) < 40:
        return {
            "macd": np.nan,
            "macd_prev": np.nan,
            "signal": np.nan,
            "signal_prev": np.nan,
            "hist": np.nan,
            "hist_prev": np.nan,
            "hist_prev2": np.nan,

            "macd_positive": False,
            "macd_cross_zero": False,
            "macd_rising": False,

            "hist_rising": False,
            "hist_rising_2": False,

            "golden_cross": False,
            "hist_negative_reducing": False,
        }

    macd, signal, hist = calculate_macd(
        df["Close"]
    )

    if len(macd) < 3:
        return {
            "macd": np.nan,
            "macd_prev": np.nan,
            "signal": np.nan,
            "signal_prev": np.nan,
            "hist": np.nan,
            "hist_prev": np.nan,
            "hist_prev2": np.nan,

            "macd_positive": False,
            "macd_cross_zero": False,
            "macd_rising": False,

            "hist_rising": False,
            "hist_rising_2": False,

            "golden_cross": False,
            "hist_negative_reducing": False,
        }

    m = float(macd.iloc[-1])
    mp = float(macd.iloc[-2])

    s = float(signal.iloc[-1])
    sp = float(signal.iloc[-2])

    h = float(hist.iloc[-1])
    hp = float(hist.iloc[-2])
    hp2 = float(hist.iloc[-3])

    return {
        "macd": m,
        "macd_prev": mp,
        "signal": s,
        "signal_prev": sp,

        "hist": h,
        "hist_prev": hp,
        "hist_prev2": hp2,

        "macd_positive": m > 0,

        "macd_cross_zero": (
            mp <= 0
            and m > 0
        ),

        "macd_rising": m > mp,

        "hist_rising": h > hp,

        "hist_rising_2": (
            h > hp
            and hp > hp2
        ),

        "golden_cross": (
            mp <= sp
            and m > s
        ),

        "hist_negative_reducing": (
            h < 0
            and h > hp
        ),
    }


# ============================================================
# 最近 MACD 突破 0 軸
# ============================================================

def get_recent_zero_cross_info(
    df,
    lookback=6,
):

    if df is None or len(df) < 40:
        return {
            "crossed": False,
            "bars_ago": None,
        }

    macd, _, _ = calculate_macd(
        df["Close"]
    )

    macd = macd.dropna()

    if len(macd) < 2:
        return {
            "crossed": False,
            "bars_ago": None,
        }

    # 現在必須仍在0軸上方
    if macd.iloc[-1] <= 0:
        return {
            "crossed": False,
            "bars_ago": None,
        }

    end = len(macd) - 1

    start = max(
        1,
        end - lookback + 1,
    )

    # 從最新往回找
    for i in range(
        end,
        start - 1,
        -1,
    ):

        previous = macd.iloc[i - 1]
        current = macd.iloc[i]

        if (
            previous <= 0
            and current > 0
        ):

            return {
                "crossed": True,
                "bars_ago": end - i,
            }

    return {
        "crossed": False,
        "bars_ago": None,
    }


# ============================================================
# 策略1 / 策略2
# MACD負值縮小 + KD
# ============================================================

def check_macd_negative_reducing_kd(
    df,
    kd_threshold=20,
):

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    if np.isnan(macd_state["hist"]):
        return False, {
            "macd_state": macd_state,
            "kd_state": kd_state,
        }

    # --------------------------------------------------------
    # MACD負值區間逐漸縮小
    # --------------------------------------------------------

    hist_condition = (
        macd_state["hist"] < 0
        and (
            macd_state["hist"] > macd_state["hist_prev"]
            or macd_state["hist_rising_2"]
        )
    )

    # --------------------------------------------------------
    # 或 MACD本身仍在負值，但開始上升
    # --------------------------------------------------------

    macd_condition = (
        macd_state["macd"] < 0
        and macd_state["macd_rising"]
    )

    macd_ok = (
        hist_condition
        or macd_condition
    )

    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    return (
        macd_ok and kd_ok
    ), {
        "macd_state": macd_state,
        "kd_state": kd_state,
        "macd_ok": macd_ok,
        "kd_ok": kd_ok,
    }


# ============================================================
# 策略3 / 4 / 5
# MACD > 0 + KD
# 可選擇要求近期突破0軸
# ============================================================

def check_macd_above_zero_kd(
    df,
    kd_threshold=20,
    zero_cross_required=False,
    zero_cross_lookback=6,
):

    macd_state = get_macd_state(df)
    kd_state = get_kd_state(df)

    macd_positive = (
        macd_state["macd_positive"]
    )

    kd_ok = (
        not np.isnan(kd_state["k"])
        and not np.isnan(kd_state["d"])
        and kd_state["k"] > kd_threshold
        and kd_state["d"] > kd_threshold
    )

    zero_cross_info = get_recent_zero_cross_info(
        df,
        zero_cross_lookback,
    )

    if zero_cross_required:

        macd_ok = (
            macd_positive
            and zero_cross_info["crossed"]
        )

    else:

        macd_ok = macd_positive

    return (
        macd_ok and kd_ok
    ), {
        "macd_state": macd_state,
        "kd_state": kd_state,
        "zero_cross": zero_cross_info,
        "macd_ok": macd_ok,
        "kd_ok": kd_ok,
    }


# ============================================================
# 60分K最終觸發
#
# 綠柱縮小：
# hist < 0
# hist > hist_prev
# hist_prev > hist_prev2
#
# 綠柱 → 紅柱：
# hist_prev < 0
# hist >= 0
# ============================================================

def check_60m_trigger(df):

    macd_state = get_macd_state(df)

    if np.isnan(macd_state["hist"]):
        return False, {
            "green_shrinking": False,
            "green_to_red": False,
            "macd_state": macd_state,
        }

    hist = macd_state["hist"]
    hist_prev = macd_state["hist_prev"]
    hist_prev2 = macd_state["hist_prev2"]

    # --------------------------------------------------------
    # 綠柱縮小
    # --------------------------------------------------------

    green_shrinking = (
        hist < 0
        and hist > hist_prev
        and hist_prev > hist_prev2
    )

    # --------------------------------------------------------
    # 綠柱轉紅柱
    # --------------------------------------------------------

    green_to_red = (
        hist_prev < 0
        and hist >= 0
    )

    trigger = (
        green_shrinking
        or green_to_red
    )

    return trigger, {
        "green_shrinking": green_shrinking,
        "green_to_red": green_to_red,
        "macd_state": macd_state,
    }


# ============================================================
# 日K強度
# ============================================================

def calculate_daily_strength(df):

    if df is None or len(df) < 120:
        return 0

    close = df["Close"]
    volume = df["Volume"]

    current_close = close.iloc[-1]

    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma120 = close.rolling(120).mean().iloc[-1]

    score = 0

    # --------------------------------------------------------
    # 價格位置
    # --------------------------------------------------------

    if current_close > ma5:
        score += 5

    if current_close > ma20:
        score += 10

    if current_close > ma60:
        score += 10

    if current_close > ma120:
        score += 10

    # --------------------------------------------------------
    # 均線排列
    # --------------------------------------------------------

    if (
        ma5 > ma20
        and ma20 > ma60
    ):
        score += 15

    if (
        ma20 > ma60
        and ma60 > ma120
    ):
        score += 10

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_state = get_macd_state(df)

    if macd_state["macd_positive"]:
        score += 10

    if macd_state["macd_rising"]:
        score += 5

    if macd_state["golden_cross"]:
        score += 5

    # --------------------------------------------------------
    # KD
    # --------------------------------------------------------

    kd_state = get_kd_state(df)

    if (
        not np.isnan(kd_state["k"])
        and kd_state["k"] > 50
    ):
        score += 5

    if kd_state["golden_cross"]:
        score += 5

    # --------------------------------------------------------
    # 成交量
    # --------------------------------------------------------

    if len(volume) >= 21:

        avg_volume = (
            volume.iloc[-21:-1]
            .mean()
        )

        if (
            avg_volume > 0
            and volume.iloc[-1]
            > avg_volume
        ):
            score += 5

    return int(min(score, 100))


# ============================================================
# S1 ~ S6 + 60K Trigger
# ============================================================

def scan_stock(
    ticker,
    daily_df,
    weekly_df,
    monthly_df,
    intraday_30_df=None,
    intraday_60_df=None,
):

    code = get_ticker_code(ticker)

    name = DYNAMIC_STOCK_NAMES.get(
        code,
        "",
    )

    label = get_stock_label(
        code,
        name,
    )

    result = {
        "ticker": ticker,
        "code": code,
        "name": name,
        "label": label,

        "s1": False,
        "s2": False,
        "s3": False,
        "s4": False,
        "s5": False,
        "s6": False,

        "m60_trigger": False,
        "m60_green_shrinking": False,
        "m60_green_to_red": False,

        "daily_strength": 0,
        "score": 0,
        "grade": "C",

        "price": np.nan,

        "details": {},
    }

    if daily_df is None or daily_df.empty:
        return result

    # ========================================================
    # 最新價格
    # ========================================================

    try:
        result["price"] = float(
            daily_df["Close"].iloc[-1]
        )
    except Exception:
        pass

    # ========================================================
    # 日K強度
    # ========================================================

    result["daily_strength"] = (
        calculate_daily_strength(
            daily_df
        )
    )

    # ========================================================
    # 策略1
    # 30K
    # ========================================================

    if (
        intraday_30_df is not None
        and not intraday_30_df.empty
    ):

        s1, d1 = check_macd_negative_reducing_kd(
            intraday_30_df,
            INTRADAY_KD_THRESHOLD,
        )

        result["s1"] = s1
        result["details"]["s1"] = d1

    # ========================================================
    # 策略2
    # 60K
    # ========================================================

    if (
        intraday_60_df is not None
        and not intraday_60_df.empty
    ):

        s2, d2 = check_macd_negative_reducing_kd(
            intraday_60_df,
            INTRADAY_KD_THRESHOLD,
        )

        result["s2"] = s2
        result["details"]["s2"] = d2

        # ----------------------------------------------------
        # 60K 最終觸發
        # ----------------------------------------------------

        trigger, trigger_info = (
            check_60m_trigger(
                intraday_60_df
            )
        )

        result["m60_trigger"] = trigger

        result["m60_green_shrinking"] = (
            trigger_info["green_shrinking"]
        )

        result["m60_green_to_red"] = (
            trigger_info["green_to_red"]
        )

        result["details"]["m60_trigger"] = (
            trigger_info
        )

    # ========================================================
    # 策略3
    # 日K
    #
    # 日K：
    # MACD > 0
    # KD > 20
    # ========================================================

    s3, d3 = check_macd_above_zero_kd(
        daily_df,
        DAILY_KD_THRESHOLD,
        zero_cross_required=False,
    )

    result["s3"] = s3
    result["details"]["s3"] = d3

    # ========================================================
    # 策略4
    # 週K
    #
    # MACD近期突破0軸
    # 且目前仍 > 0
    # KD > 50
    # ========================================================

    s4 = False
    d4 = {}

    if (
        weekly_df is not None
        and not weekly_df.empty
    ):

        s4, d4 = check_macd_above_zero_kd(
            weekly_df,
            WEEKLY_KD_THRESHOLD,
            zero_cross_required=True,
            zero_cross_lookback=(
                WEEKLY_ZERO_CROSS_LOOKBACK
            ),
        )

    result["s4"] = s4
    result["details"]["s4"] = d4

    # ========================================================
    # 策略5
    # 月K
    #
    # MACD近期突破0軸
    # 且目前仍 > 0
    # KD > 50
    # ========================================================

    s5 = False
    d5 = {}

    if (
        monthly_df is not None
        and not monthly_df.empty
    ):

        s5, d5 = check_macd_above_zero_kd(
            monthly_df,
            MONTHLY_KD_THRESHOLD,
            zero_cross_required=True,
            zero_cross_lookback=(
                MONTHLY_ZERO_CROSS_LOOKBACK
            ),
        )

    result["s5"] = s5
    result["details"]["s5"] = d5

    # ========================================================
    # 策略6
    #
    # 日K + 週K + 月K
    #
    # 三週期同時成立
    # ========================================================

    result["s6"] = (
        result["s3"]
        and result["s4"]
        and result["s5"]
    )

    # ========================================================
    # 分數
    # ========================================================

    result["score"] = calculate_strategy_score(
        result
    )

    result["grade"] = get_grade(
        result["score"]
    )

    return result


# ============================================================
# 策略分數
#
# 注意：
# 分數只做「排序」
# 不作為策略成立條件
# ============================================================

def calculate_strategy_score(result):

    score = 0

    # --------------------------------------------------------
    # 原策略權重
    # --------------------------------------------------------

    if result["s1"]:
        score += 10

    if result["s2"]:
        score += 10

    if result["s3"]:
        score += 15

    if result["s4"]:
        score += 15

    if result["s5"]:
        score += 15

    # --------------------------------------------------------
    # S6：
    # 日 + 週 + 月共振
    # --------------------------------------------------------

    if result["s6"]:
        score += 20

    # --------------------------------------------------------
    # MACD零軸突破加分
    # --------------------------------------------------------

    for key in [
        "s3",
        "s4",
        "s5",
    ]:

        details = result["details"].get(
            key,
            {}
        )

        macd_state = details.get(
            "macd_state",
            {}
        )

        if macd_state.get(
            "macd_cross_zero",
            False
        ):
            score += 2

        zero_cross = details.get(
            "zero_cross",
            {}
        )

        if zero_cross.get(
            "crossed",
            False
        ):
            score += 2

    # --------------------------------------------------------
    # KD黃金交叉
    # --------------------------------------------------------

    for key in [
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
    ]:

        details = result["details"].get(
            key,
            {}
        )

        kd_state = details.get(
            "kd_state",
            {}
        )

        if kd_state.get(
            "golden_cross",
            False
        ):
            score += 2

    # --------------------------------------------------------
    # 60K最終觸發
    # --------------------------------------------------------

    if result["m60_green_shrinking"]:
        score += 5

    if result["m60_green_to_red"]:
        score += 10

    # --------------------------------------------------------
    # S6 + 60K
    #
    # 這是整套策略最重要的組合
    # --------------------------------------------------------

    if (
        result["s6"]
        and result["m60_green_shrinking"]
    ):
        score += 10

    if (
        result["s6"]
        and result["m60_green_to_red"]
    ):
        score += 15

    # --------------------------------------------------------
    # 限制100分
    # --------------------------------------------------------

    return int(
        min(score, 100)
    )


# ============================================================
# 評級
# ============================================================

def get_grade(score):

    if score >= 85:
        return "S"

    if score >= 70:
        return "A"

    if score >= 55:
        return "B"

    if score >= 40:
        return "C"

    return "D"


# ============================================================
# 建立日K篩選池
#
# 重點：
# 優先把 S6 多週期共振股票送進 30K / 60K
# ============================================================

def build_intraday_scan_pool(
    tickers,
    daily_data,
    weekly_data,
    monthly_data,
):

    print("")
    print("=" * 70)
    print("🔎 建立 30K / 60K 優先掃描池")
    print("=" * 70)

    candidates = []

    for ticker in tickers:

        daily_df = daily_data.get(ticker)

        if (
            daily_df is None
            or len(daily_df) < 120
        ):
            continue

        try:

            # ------------------------------------------------
            # 流動性
            # ------------------------------------------------

            volume = daily_df["Volume"]

            if len(volume) < 21:
                continue

            avg_volume = (
                volume.iloc[-21:]
                .mean()
            )

            if (
                avg_volume
                < MIN_AVG_VOLUME_20
            ):
                continue

            # ------------------------------------------------
            # 價格 > MA20
            # ------------------------------------------------

            close = daily_df["Close"]

            ma20 = (
                close.rolling(20)
                .mean()
                .iloc[-1]
            )

            current_price = (
                close.iloc[-1]
            )

            if current_price <= ma20:
                continue

            # ------------------------------------------------
            # 日K強度
            # ------------------------------------------------

            strength = (
                calculate_daily_strength(
                    daily_df
                )
            )

            # ------------------------------------------------
            # 先判斷 S3
            # ------------------------------------------------

            s3, _ = check_macd_above_zero_kd(
                daily_df,
                DAILY_KD_THRESHOLD,
                zero_cross_required=False,
            )

            # ------------------------------------------------
            # S4
            # ------------------------------------------------

            weekly_df = weekly_data.get(
                ticker
            )

            s4 = False

            if (
                weekly_df is not None
                and not weekly_df.empty
            ):

                s4, _ = check_macd_above_zero_kd(
                    weekly_df,
                    WEEKLY_KD_THRESHOLD,
                    zero_cross_required=True,
                    zero_cross_lookback=(
                        WEEKLY_ZERO_CROSS_LOOKBACK
                    ),
                )

            # ------------------------------------------------
            # S5
            # ------------------------------------------------

            monthly_df = monthly_data.get(
                ticker
            )

            s5 = False

            if (
                monthly_df is not None
                and not monthly_df.empty
            ):

                s5, _ = check_macd_above_zero_kd(
                    monthly_df,
                    MONTHLY_KD_THRESHOLD,
                    zero_cross_required=True,
                    zero_cross_lookback=(
                        MONTHLY_ZERO_CROSS_LOOKBACK
                    ),
                )

            # ------------------------------------------------
            # S6
            # ------------------------------------------------

            s6 = (
                s3
                and s4
                and s5
            )

            candidates.append({
                "ticker": ticker,
                "strength": strength,
                "s3": s3,
                "s4": s4,
                "s5": s5,
                "s6": s6,
            })

        except Exception:
            continue

    if not candidates:
        print("⚠️ 沒有符合條件的日K股票")
        return []

    # ========================================================
    # 排序
    #
    # 第一優先：
    # S6
    #
    # 第二：
    # S5
    #
    # 第三：
    # S4
    #
    # 第四：
    # S3
    #
    # 第五：
    # 日K強度
    # ========================================================

    candidates.sort(
        key=lambda x: (
            int(x["s6"]),
            int(x["s5"]),
            int(x["s4"]),
            int(x["s3"]),
            x["strength"],
        ),
        reverse=True,
    )

    selected = candidates[
        :INTRADAY_SCAN_LIMIT
    ]

    s6_count = sum(
        1
        for x in selected
        if x["s6"]
    )

    print(
        f"✅ 建立掃描池：{len(selected)} 檔"
    )

    print(
        f"🔥 其中策略六共振：{s6_count} 檔"
    )

    return [
        x["ticker"]
        for x in selected
    ]


# ============================================================
# Telegram
# ============================================================

def send_telegram_message(
    message,
):

    bot_token = os.getenv(
        "TG_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TG_CHAT_ID"
    )

    if not bot_token or not chat_id:

        print(
            "⚠️ 未設定 TG_BOT_TOKEN / TG_CHAT_ID"
        )

        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=20,
        )

        response.raise_for_status()

        print("✅ Telegram 發送成功")

        return True

    except Exception as e:

        print(
            f"❌ Telegram 發送失敗：{e}"
        )

        return False


# ============================================================
# Telegram 報告
# ============================================================

def build_telegram_report(
    results,
):

    if not results:
        return (
            "📊 <b>台股多週期策略掃描</b>\n\n"
            "今日沒有符合條件的股票。"
        )

    # ========================================================
    # 排序
    #
    # S6
    # + 60K綠轉紅
    # + 60K綠柱縮小
    # + score
    # ========================================================

    results_sorted = sorted(
        results,
        key=lambda x: (
            int(x["s6"]),
            int(x["m60_green_to_red"]),
            int(x["m60_green_shrinking"]),
            x["score"],
            x["daily_strength"],
        ),
        reverse=True,
    )

    lines = []

    now = datetime.now(
        timezone(
            timedelta(hours=8)
        )
    )

    lines.append(
        f"📊 <b>台股多週期策略掃描</b>"
    )

    lines.append(
        f"版本：<b>{VERSION}</b>"
    )

    lines.append(
        f"時間：{now.strftime('%Y-%m-%d %H:%M')}"
    )

    lines.append("")

    # ========================================================
    # 統計
    # ========================================================

    s1_list = [
        x for x in results
        if x["s1"]
    ]

    s2_list = [
        x for x in results
        if x["s2"]
    ]

    s3_list = [
        x for x in results
        if x["s3"]
    ]

    s4_list = [
        x for x in results
        if x["s4"]
    ]

    s5_list = [
        x for x in results
        if x["s5"]
    ]

    s6_list = [
        x for x in results
        if x["s6"]
    ]

    green_shrink_list = [
        x for x in results
        if x["m60_green_shrinking"]
    ]

    green_red_list = [
        x for x in results
        if x["m60_green_to_red"]
    ]

    lines.append(
        "📈 <b>掃描統計</b>"
    )

    lines.append(
        f"策略一：{len(s1_list)} 檔"
    )

    lines.append(
        f"策略二：{len(s2_list)} 檔"
    )

    lines.append(
        f"策略三：{len(s3_list)} 檔"
    )

    lines.append(
        f"策略四：{len(s4_list)} 檔"
    )

    lines.append(
        f"策略五：{len(s5_list)} 檔"
    )

    lines.append(
        f"🔥 策略六共振：{len(s6_list)} 檔"
    )

    lines.append(
        f"🟢 60K綠柱縮小：{len(green_shrink_list)} 檔"
    )

    lines.append(
        f"🔴 60K綠→紅：{len(green_red_list)} 檔"
    )

    # ========================================================
    # 最重要：
    # S6 + 60K 綠→紅
    # ========================================================

    strongest = [
        x for x in results
        if (
            x["s6"]
            and x["m60_green_to_red"]
        )
    ]

    if strongest:

        lines.append("")
        lines.append(
            "🔥 <b>最高優先："
            "策略六 + 60K 綠柱→紅柱</b>"
        )

        strongest = sorted(
            strongest,
            key=lambda x: (
                x["score"],
                x["daily_strength"],
            ),
            reverse=True,
        )

        for i, item in enumerate(
            strongest[:20],
            1,
        ):

            price = item["price"]

            lines.append(
                f"{i}. <b>"
                f"{escape_html(item['label'])}"
                f"</b> "
                f"{price:.2f} "
                f"｜{item['score']}分"
            )

    # ========================================================
    # S6 + 60K 綠柱縮小
    # ========================================================

    second_priority = [
        x for x in results
        if (
            x["s6"]
            and x["m60_green_shrinking"]
            and not x["m60_green_to_red"]
        )
    ]

    if second_priority:

        lines.append("")
        lines.append(
            "🟢 <b>第二優先："
            "策略六 + 60K 綠柱縮小</b>"
        )

        second_priority = sorted(
            second_priority,
            key=lambda x: (
                x["score"],
                x["daily_strength"],
            ),
            reverse=True,
        )

        for i, item in enumerate(
            second_priority[:20],
            1,
        ):

            lines.append(
                f"{i}. <b>"
                f"{escape_html(item['label'])}"
                f"</b> "
                f"{item['price']:.2f} "
                f"｜{item['score']}分"
            )

    # ========================================================
    # 策略六全部
    # ========================================================

    if s6_list:

        lines.append("")
        lines.append(
            "🔥 <b>策略六："
            "日／週／月多週期共振</b>"
        )

        s6_sorted = sorted(
            s6_list,
            key=lambda x: (
                x["m60_green_to_red"],
                x["m60_green_shrinking"],
                x["score"],
                x["daily_strength"],
            ),
            reverse=True,
        )

        for i, item in enumerate(
            s6_sorted[:30],
            1,
        ):

            trigger = ""

            if item["m60_green_to_red"]:
                trigger = " 🔴綠→紅"

            elif item["m60_green_shrinking"]:
                trigger = " 🟢綠柱縮小"

            lines.append(
                f"{i}. <b>"
                f"{escape_html(item['label'])}"
                f"</b> "
                f"{item['price']:.2f}"
                f"｜{item['score']}分"
                f"{trigger}"
            )

    # ========================================================
    # 策略三
    # ========================================================

    if s3_list:

        lines.append("")
        lines.append(
            "📅 <b>策略三：日K MACD > 0 + KD > 20</b>"
        )

        for i, item in enumerate(
            sorted(
                s3_list,
                key=lambda x: (
                    x["score"],
                    x["daily_strength"],
                ),
                reverse=True,
            )[:20],
            1,
        ):

            lines.append(
                f"{i}. "
                f"{escape_html(item['label'])} "
                f"{item['price']:.2f}"
            )

    # ========================================================
    # 策略四
    # ========================================================

    if s4_list:

        lines.append("")
        lines.append(
            "📆 <b>策略四：週K MACD突破0軸 + KD > 50</b>"
        )

        for i, item in enumerate(
            sorted(
                s4_list,
                key=lambda x: x["score"],
                reverse=True,
            )[:20],
            1,
        ):

            lines.append(
                f"{i}. "
                f"{escape_html(item['label'])} "
                f"{item['price']:.2f}"
            )

    # ========================================================
    # 策略五
    # ========================================================

    if s5_list:

        lines.append("")
        lines.append(
            "📅 <b>策略五：月K MACD突破0軸 + KD > 50</b>"
        )

        for i, item in enumerate(
            sorted(
                s5_list,
                key=lambda x: x["score"],
                reverse=True,
            )[:20],
            1,
        ):

            lines.append(
                f"{i}. "
                f"{escape_html(item['label'])} "
                f"{item['price']:.2f}"
            )

    # ========================================================
    # 60K綠柱縮小
    # ========================================================

    if green_shrink_list:

        lines.append("")
        lines.append(
            "🟢 <b>60分K：MACD綠柱縮小</b>"
        )

        for i, item in enumerate(
            sorted(
                green_shrink_list,
                key=lambda x: (
                    x["s6"],
                    x["score"],
                ),
                reverse=True,
            )[:20],
            1,
        ):

            lines.append(
                f"{i}. "
                f"{escape_html(item['label'])} "
                f"{item['price']:.2f}"
            )

    # ========================================================
    # 60K綠轉紅
    # ========================================================

    if green_red_list:

        lines.append("")
        lines.append(
            "🔴 <b>60分K：MACD綠柱→紅柱</b>"
        )

        for i, item in enumerate(
            sorted(
                green_red_list,
                key=lambda x: (
                    x["s6"],
                    x["score"],
                ),
                reverse=True,
            )[:20],
            1,
        ):

            lines.append(
                f"{i}. "
                f"{escape_html(item['label'])} "
                f"{item['price']:.2f}"
            )

    # ========================================================
    # Top 20 綜合
    # ========================================================

    lines.append("")
    lines.append(
        "🏆 <b>Top 20 綜合排名</b>"
    )

    for i, item in enumerate(
        results_sorted[:20],
        1,
    ):

        flags = []

        if item["s6"]:
            flags.append("S6🔥")

        if item["s5"]:
            flags.append("S5")

        if item["s4"]:
            flags.append("S4")

        if item["s3"]:
            flags.append("S3")

        if item["s2"]:
            flags.append("S2")

        if item["s1"]:
            flags.append("S1")

        if item["m60_green_to_red"]:
            flags.append("60K轉紅")

        elif item["m60_green_shrinking"]:
            flags.append("60K縮柱")

        flag_text = " ".join(flags)

        lines.append(
            f"{i}. <b>"
            f"{escape_html(item['label'])}"
            f"</b> "
            f"{item['price']:.2f}"
            f"｜{item['score']}分 "
            f"{flag_text}"
        )

    # ========================================================
    # 策略說明
    # ========================================================

    lines.append("")
    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>本版核心邏輯</b>"
    )

    lines.append(
        "月K MACD突破0軸"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "週K MACD突破0軸"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "日K MACD > 0"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "60分K MACD綠柱縮小"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "綠柱 → 紅柱"
    )

    lines.append(
        "↓"
    )

    lines.append(
        "🔥 最終多週期共振"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    return "\n".join(lines)


# ============================================================
# 主程式
# ============================================================

def main():

    print("")
    print("=" * 70)
    print(
        f"🚀 Taiwan Stock Scanner {VERSION}"
    )
    print("=" * 70)

    # ========================================================
    # 台灣時間
    # ========================================================

    taiwan_tz = timezone(
        timedelta(hours=8)
    )

    now = datetime.now(
        taiwan_tz
    )

    print(
        f"🕐 台灣時間："
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ========================================================
    # 股票清單
    # ========================================================

    tickers = (
        fetch_all_taiwan_market_tickers()
    )

    if not tickers:

        print(
            "❌ 沒有股票清單，程式結束"
        )

        return

    # ========================================================
    # 日K
    # ========================================================

    daily_data = safe_download_yf(
        tickers,
        period="1y",
        interval="1d",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    if not daily_data:

        print(
            "❌ 日K資料取得失敗"
        )

        return

    # ========================================================
    # 週K
    # ========================================================

    weekly_data = safe_download_yf(
        tickers,
        period="2y",
        interval="1wk",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    # ========================================================
    # 月K
    # ========================================================

    monthly_data = safe_download_yf(
        tickers,
        period="5y",
        interval="1mo",
        chunk_size=DAILY_CHUNK_SIZE,
    )

    # ========================================================
    # 建立30K / 60K掃描池
    #
    # 這裡已經優先把策略六送進去
    # ========================================================

    intraday_pool = (
        build_intraday_scan_pool(
            tickers,
            daily_data,
            weekly_data,
            monthly_data,
        )
    )

    # ========================================================
    # 30分K
    # ========================================================

    intraday_30_data = {}

    if intraday_pool:

        intraday_30_data = safe_download_yf(
            intraday_pool,
            period="1mo",
            interval="30m",
            chunk_size=INTRADAY_CHUNK_SIZE,
        )

    # ========================================================
    # 60分K
    # ========================================================

    intraday_60_data = {}

    if intraday_pool:

        intraday_60_data = safe_download_yf(
            intraday_pool,
            period="1mo",
            interval="60m",
            chunk_size=INTRADAY_CHUNK_SIZE,
        )

    # ========================================================
    # 正式掃描
    # ========================================================

    print("")
    print("=" * 70)
    print("🔍 正式掃描策略")
    print("=" * 70)

    results = []

    for i, ticker in enumerate(
        tickers,
        1,
    ):

        daily_df = daily_data.get(
            ticker
        )

        if (
            daily_df is None
            or daily_df.empty
        ):
            continue

        weekly_df = weekly_data.get(
            ticker
        )

        monthly_df = monthly_data.get(
            ticker
        )

        df30 = intraday_30_data.get(
            ticker
        )

        df60 = intraday_60_data.get(
            ticker
        )

        try:

            result = scan_stock(
                ticker=ticker,
                daily_df=daily_df,
                weekly_df=weekly_df,
                monthly_df=monthly_df,
                intraday_30_df=df30,
                intraday_60_df=df60,
            )

            # ------------------------------------------------
            # 有任何策略或60K觸發才保留
            # ------------------------------------------------

            active = (
                result["s1"]
                or result["s2"]
                or result["s3"]
                or result["s4"]
                or result["s5"]
                or result["s6"]
                or result["m60_trigger"]
            )

            if active:
                results.append(result)

        except Exception as e:

            print(
                f"⚠️ {ticker}掃描失敗：{e}"
            )

    # ========================================================
    # 排序
    # ========================================================

    results.sort(
        key=lambda x: (
            int(x["s6"]),
            int(x["m60_green_to_red"]),
            int(x["m60_green_shrinking"]),
            x["score"],
            x["daily_strength"],
        ),
        reverse=True,
    )

    # ========================================================
    # 統計
    # ========================================================

    print("")
    print("=" * 70)
    print("📊 掃描完成")
    print("=" * 70)

    print(
        f"符合條件：{len(results)} 檔"
    )

    print(
        f"策略一："
        f"{sum(x['s1'] for x in results)}"
    )

    print(
        f"策略二："
        f"{sum(x['s2'] for x in results)}"
    )

    print(
        f"策略三："
        f"{sum(x['s3'] for x in results)}"
    )

    print(
        f"策略四："
        f"{sum(x['s4'] for x in results)}"
    )

    print(
        f"策略五："
        f"{sum(x['s5'] for x in results)}"
    )

    print(
        f"🔥 策略六："
        f"{sum(x['s6'] for x in results)}"
    )

    print(
        f"🟢 60K綠柱縮小："
        f"{sum(x['m60_green_shrinking'] for x in results)}"
    )

    print(
        f"🔴 60K綠→紅："
        f"{sum(x['m60_green_to_red'] for x in results)}"
    )

    # ========================================================
    # Console Top 20
    # ========================================================

    print("")
    print(
        "🏆 Top 20"
    )

    print("-" * 70)

    for i, item in enumerate(
        results[:20],
        1,
    ):

        flags = []

        if item["s6"]:
            flags.append("S6🔥")

        if item["s5"]:
            flags.append("S5")

        if item["s4"]:
            flags.append("S4")

        if item["s3"]:
            flags.append("S3")

        if item["s2"]:
            flags.append("S2")

        if item["s1"]:
            flags.append("S1")

        if item["m60_green_to_red"]:
            flags.append("60K→紅")

        elif item["m60_green_shrinking"]:
            flags.append("60K縮柱")

        print(
            f"{i:02d}. "
            f"{item['label']:<20} "
            f"{item['price']:>8.2f} "
            f"Score={item['score']:>3} "
            f"{' '.join(flags)}"
        )

    # ========================================================
    # Telegram
    # ========================================================

    report = build_telegram_report(
        results
    )

    print("")
    print(
        "📨 發送 Telegram..."
    )

    send_telegram_message(
        report
    )

    print("")
    print("=" * 70)
    print("✅ 程式執行完成")
    print("=" * 70)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
