#!/usr/bin/env python3
"""Daily RSI(14) cross-below-30 and Bollinger Band (20, SMA, close, ±2σ)
band-cross alerts, notified via LINE Messaging API broadcast.

Run with --market jp at 16:00 JST (after the TSE close) and
--market us at 06:00 JST (after the US market close), so each run only
fetches the tickers relevant to that market.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
OUT_DIR = SCRIPT_DIR / ".out"

CHART_LOOKBACK_DAYS = 90

RSI_PERIOD = 14
RSI_THRESHOLD = 30

BB_PERIOD = 20
BB_NUM_STD = 2

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_NEAR_CROSS_RATIO = 0.3  # histogram within 30% of its recent average magnitude

TICKERS_JP = [
    "9984.T",   # ソフトバンク
    "285A.T",   # キオクシア
    "6857.T",   # アドバンテスト
    "4062.T",   # イビデン
    "6758.T",   # ソニー
    "8035.T",   # 東京エレクトロン
    "7735.T",   # SCREEN
    "5803.T",   # フジクラ
    "8306.T",   # 三菱UFJ
    "6702.T",   # 富士通
    "1551.T",   # 1551
    "^N225",    # NI225 (日経225)
    "1306.T",   # TOPIX / 東証プライム市場指数(I0500)の代用ETF
    "2516.T",   # MOS (東証グロース市場250指数 ETF)
    "5401.T", "4107.T", "5801.T", "6954.T", "5706.T", "7203.T", "7261.T",
    "5713.T", "4063.T", "7012.T", "7832.T", "9697.T", "6460.T", "7011.T",
    "485A.T", "3692.T", "6532.T", "6701.T", "4980.T", "9766.T", "7746.T",
    "3402.T", "6323.T", "7974.T", "7433.T", "278A.T", "3103.T", "5901.T",
    "5016.T", "3110.T", "1963.T", "6301.T", "7272.T", "5844.T", "4502.T",
    "4568.T", "8411.T", "9983.T", "7550.T", "2702.T", "3563.T", "4443.T",
    "6762.T", "6981.T", "5333.T", "6501.T", "6674.T", "7014.T", "8031.T",
    "6269.T", "7003.T", "9104.T", "1605.T", "7267.T",
    "3498.T", "7013.T", "6861.T", "6098.T", "8001.T", "8053.T", "8002.T",
    "6146.T", "3863.T", "3864.T", "7912.T", "7751.T", "8766.T", "9031.T",
    "9042.T", "9021.T", "9684.T", "3156.T", "6254.T", "3436.T", "4704.T",
    "4519.T", "4202.T", "4506.T", "5726.T", "6330.T", "3431.T", "5711.T",
]

TICKERS_US = [
    "BTC-USD",
    "MSTR", "GEMI", "NVDA", "MRVL", "TER", "INTC", "TSM", "ARM",
    "ASML", "TSLA", "ORCL", "LITE", "SNDK", "SKHY", "SMSN.L",
    "MU", "CAT", "QCOM", "XOM", "CVX", "FORM", "GEV", "BAC", "CRM",
    "^GSPC",      # SP500
    "^SOX",       # SOX
    "^DJI",       # DJI
    "^NDX",       # NAS100
    "^RUT",       # RUT
    "^HSI",       # HSI
    "^KS11",      # KOSPI
    "^GDAXI",     # DAX
    "GC=F",       # GOLD
    "SI=F",       # SILVER
    "PL=F",       # XPTUSD
    "HG=F",       # COPPER
    "CL=F",       # USOIL (CL1!と同一のため一本化)
    "ETH-USD",    # ETHUSD
    "^STOXX50E",  # ESXEUR
    "AVGO", "AMAT", "LRCX", "TXN", "QRVO", "RMBS", "VOOG", "VOOV",
    "LMT", "BA", "LHX", "OXY", "CRWV", "IBM", "GOOG", "NFLX", "AMZN",
    "META", "MSFT", "AAPL", "PLTR", "SCCO", "ERO", "BKSY", "SPCX",
    "AREC", "USAR", "IPX", "COIN", "CLSK", "MARA", "PM", "MO", "BTI",
    "PFE", "LLY", "MRNA", "JNJ",
    "ROKU", "U", "PINS",
]

MARKETS = {
    "jp": ("日本株", TICKERS_JP),
    "us": ("米国株・指数・コモディティ", TICKERS_US),
}


def load_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    # Environment variables (e.g. GitHub Actions secrets) take precedence over .env.
    for key in ("LINE_CHANNEL_ACCESS_TOKEN", "IMGBB_API_KEY"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def load_credentials():
    env = load_env()
    token = env.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    imgbb_key = env.get("IMGBB_API_KEY", "").strip()
    if not token:
        sys.exit("LINE_CHANNEL_ACCESS_TOKEN missing (set in .env or as an environment variable)")
    if not imgbb_key:
        sys.exit("IMGBB_API_KEY missing (set in .env or as an environment variable)")
    try:
        token.encode("ascii")
    except UnicodeEncodeError as e:
        bad = [f"U+{ord(c):04X}" for c in token if ord(c) > 127]
        sys.exit(
            f"LINE_CHANNEL_ACCESS_TOKEN contains non-ASCII characters (length={len(token)}, "
            f"offending codepoints={bad[:5]}). The secret value is likely corrupted by copy-paste "
            f"(stray whitespace, smart quotes, or zero-width characters) — re-copy it directly from "
            f"the LINE Developers console and re-save the secret. Original error: {e}"
        )
    return token, imgbb_key


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_bbands(close: pd.Series, period: int = BB_PERIOD, num_std: float = BB_NUM_STD):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower


def compute_macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def check_ticker(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        return None, f"{ticker}: fetch error ({e})"
    min_bars = max(RSI_PERIOD, BB_PERIOD) + 2
    if df is None or df.empty or len(df) < min_bars:
        return None, f"{ticker}: insufficient data"
    close = df["Close"].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    rsi = compute_rsi(close)
    upper, lower = compute_bbands(close)
    _, _, hist = compute_macd(close)
    if (
        len(rsi) < 2 or pd.isna(rsi.iloc[-1]) or pd.isna(rsi.iloc[-2])
        or pd.isna(upper.iloc[-1]) or pd.isna(upper.iloc[-2])
        or len(hist) < 3 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-2])
    ):
        return None, f"{ticker}: insufficient indicator data"

    prev_rsi, curr_rsi = float(rsi.iloc[-2]), float(rsi.iloc[-1])
    prev_close, curr_close = float(close.iloc[-2]), float(close.iloc[-1])
    prev_upper, curr_upper = float(upper.iloc[-2]), float(upper.iloc[-1])
    prev_lower, curr_lower = float(lower.iloc[-2]), float(lower.iloc[-1])
    prev_hist, curr_hist = float(hist.iloc[-2]), float(hist.iloc[-1])

    rsi_crossed_down = prev_rsi >= RSI_THRESHOLD and curr_rsi < RSI_THRESHOLD
    bb_crossed_up = prev_close <= prev_upper and curr_close > curr_upper
    bb_crossed_down = prev_close >= prev_lower and curr_close < curr_lower
    below_lower_now = curr_close < curr_lower

    macd_golden_cross = prev_hist <= 0 and curr_hist > 0
    hist_scale = float(hist.iloc[-20:].abs().mean())
    macd_near_cross = (
        not macd_golden_cross
        and curr_hist < 0
        and curr_hist > prev_hist
        and hist_scale > 0
        and abs(curr_hist) < MACD_NEAR_CROSS_RATIO * hist_scale
    )

    return {
        "ticker": ticker,
        "date": close.index[-1].strftime("%Y-%m-%d"),
        "prev_rsi": round(prev_rsi, 1),
        "curr_rsi": round(curr_rsi, 1),
        "close": round(curr_close, 2),
        "upper": round(curr_upper, 2),
        "lower": round(curr_lower, 2),
        "rsi_crossed_down": rsi_crossed_down,
        "bb_crossed_up": bb_crossed_up,
        "bb_crossed_down": bb_crossed_down,
        "below_lower_now": below_lower_now,
        "macd_golden_cross": macd_golden_cross,
        "macd_near_cross": macd_near_cross,
    }, None


def render_chart(ticker: str, out_path: Path, lookback: int = CHART_LOOKBACK_DAYS):
    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    close = df["Close"].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    rsi = compute_rsi(close)
    sma = close.rolling(BB_PERIOD).mean()
    upper, lower = compute_bbands(close)
    macd_line, signal_line, hist = compute_macd(close)

    c, s, u, l, r, m, sg, h = (
        x.iloc[-lookback:] for x in (close, sma, upper, lower, rsi, macd_line, signal_line, hist)
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    ax1.plot(c.index, c.values, label="Close", color="#1f77b4")
    ax1.plot(s.index, s.values, label="SMA20", color="#888888", linewidth=1)
    ax1.plot(u.index, u.values, label="+2σ", color="#d62728", linewidth=1)
    ax1.plot(l.index, l.values, label="-2σ", color="#2ca02c", linewidth=1)
    ax1.fill_between(c.index, l.values, u.values, color="#cccccc", alpha=0.2)
    ax1.set_title(ticker)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(r.index, r.values, color="#9467bd", label="RSI(14)")
    ax2.axhline(30, color="#d62728", linewidth=1, linestyle="--")
    ax2.axhline(70, color="#2ca02c", linewidth=1, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in h.values]
    ax3.bar(h.index, h.values, color=bar_colors, width=1.0, label="MACD hist")
    ax3.plot(m.index, m.values, color="#1f77b4", linewidth=1, label="MACD")
    ax3.plot(sg.index, sg.values, color="#ff7f0e", linewidth=1, label="Signal")
    ax3.axhline(0, color="#888888", linewidth=0.8)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def upload_to_imgbb(api_key: str, png_path: Path) -> str:
    with open(png_path, "rb") as f:
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": api_key},
            files={"image": f},
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"imgbb upload failed: {data}")
    return data["data"]["url"]


def send_line_messages(token: str, messages: list):
    # LINE allows at most 5 message objects per broadcast call.
    for i in range(0, len(messages), 5):
        chunk = messages[i:i + 5]
        resp = requests.post(
            "https://api.line.me/v2/bot/message/broadcast",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"messages": chunk},
            timeout=15,
        )
        resp.raise_for_status()


def run_market(market: str, token: str, imgbb_key: str):
    label, tickers = MARKETS[market]
    rsi_hits, bb_up_hits, bb_down_hits, errors = [], [], [], []
    candidate_hits, dip_hits = [], []
    latest_date = None

    for ticker in tickers:
        result, err = check_ticker(ticker)
        if err:
            errors.append(err)
            continue
        latest_date = result["date"]
        if result["rsi_crossed_down"]:
            rsi_hits.append(result)
        if result["bb_crossed_up"]:
            bb_up_hits.append(result)
        if result["bb_crossed_down"]:
            bb_down_hits.append(result)
        if result["bb_crossed_up"] and (result["macd_golden_cross"] or result["macd_near_cross"]):
            candidate_hits.append(result)
        if result["below_lower_now"]:
            dip_hits.append(result)

    if errors:
        print("Errors:\n" + "\n".join(errors), file=sys.stderr)

    any_hits = rsi_hits or bb_up_hits or bb_down_hits or candidate_hits or dip_hits
    lines = [f"【BB日足チェック：{label}】", f"{latest_date or '?'} 日足"]

    if rsi_hits:
        lines.append("── RSI(14) 30割れ ──")
        for h in rsi_hits:
            lines.append(f"{h['ticker']}: RSI {h['prev_rsi']} → {h['curr_rsi']} (終値 {h['close']})")
    if bb_up_hits:
        lines.append("── ボリンジャーバンド +2σ 上抜け ──")
        for h in bb_up_hits:
            lines.append(f"{h['ticker']}: 終値 {h['close']} > 上限 {h['upper']}")
    if bb_down_hits:
        lines.append("── ボリンジャーバンド -2σ 下抜け ──")
        for h in bb_down_hits:
            lines.append(f"{h['ticker']}: 終値 {h['close']} < 下限 {h['lower']}")
    if candidate_hits:
        lines.append("🌟 有力銘柄候補（BB+2σ上抜け×MACD）")
        for h in candidate_hits:
            tag = "ゴールデンクロス" if h["macd_golden_cross"] else "GC接近"
            lines.append(f"{h['ticker']}: 終値 {h['close']} (MACD{tag})")
    if dip_hits:
        lines.append("📉 BUY THE DIP候補（-2σ下抜け中）")
        for h in dip_hits:
            lines.append(f"{h['ticker']}: 終値 {h['close']} < 下限 {h['lower']}")
    if not any_hits:
        lines.append("該当する銘柄はありませんでした。")

    message = "\n".join(lines)

    if not any_hits:
        print(message)
        send_line_messages(token, [{"type": "text", "text": message}])
        return

    OUT_DIR.mkdir(exist_ok=True)
    hit_tickers = []
    seen = set()
    for h in rsi_hits + bb_up_hits + bb_down_hits + candidate_hits + dip_hits:
        if h["ticker"] not in seen:
            seen.add(h["ticker"])
            hit_tickers.append(h["ticker"])

    image_urls = []
    for ticker in hit_tickers:
        safe = "".join(c if c.isalnum() else "_" for c in ticker)
        png_path = OUT_DIR / f"{market}_{safe}.png"
        try:
            render_chart(ticker, png_path)
            url = upload_to_imgbb(imgbb_key, png_path)
            image_urls.append(url)
        except Exception as e:
            print(f"chart/upload error for {ticker}: {e}", file=sys.stderr)

    print(message)
    messages = [{"type": "text", "text": message}]
    for url in image_urls:
        messages.append({"type": "image", "originalContentUrl": url, "previewImageUrl": url})
    send_line_messages(token, messages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    args = parser.parse_args()

    token, imgbb_key = load_credentials()
    run_market(args.market, token, imgbb_key)


if __name__ == "__main__":
    main()
