#Stock Technical Analysis and Trading Signal Generator - A DSA Project
# ---------- LAYER 1: DATA LAYER ----------
from dataclasses import dataclass
from datetime import date, timedelta
from collections import deque
import math
import yfinance as yf
import pandas as pd

@dataclass
class StockRecord:
    """Bản ghi OHLCV — đơn vị dữ liệu cơ bản."""
    date:   date
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int

def fetch_data(symbol: str = "VNM", days: int = 120) -> list:
    # Thử lấy dữ liệu nguyên bản trước (quốc tế)
    df = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
    
    # Nếu không có dữ liệu, thử thêm .VN cho chứng khoán Việt Nam
    if df.empty:
        df = yf.download(symbol + ".VN", period="6mo", auto_adjust=True, progress=False)
        
    if df.empty:
        raise ValueError(f"No real data found for symbol '{symbol}' on yfinance.")
        
    records = []
    # yfinance sometimes returns MultiIndex columns. Flattening them just in case.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    for idx, row in df.iterrows():
        records.append(StockRecord(
            date=idx.date(),
            open=float(row.iloc[0] if hasattr(row["Open"], "iloc") else row["Open"]),
            high=float(row.iloc[0] if hasattr(row["High"], "iloc") else row["High"]),
            low=float(row.iloc[0] if hasattr(row["Low"], "iloc") else row["Low"]),
            close=float(row.iloc[0] if hasattr(row["Close"], "iloc") else row["Close"]),
            volume=int(row.iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"])
        ))
    return records


# ---------- LAYER 2: DATA STORE & CACHE ----------
class StockDataStore:
    """
    Lưu trữ chính — O(1) access theo index.
    Dùng list[float] cho prices để tối ưu bộ nhớ và tốc độ.
    """
    def __init__(self, records: list):
        self.records     = records
        self.prices      = [r.close  for r in records]   
        self.dates       = [r.date   for r in records]
        self.highs       = [r.high   for r in records]
        self.lows        = [r.low    for r in records]
        self.volumes     = [r.volume for r in records]

    def get_price(self, idx: int) -> float:
        return self.prices[idx]

class StockCache:
    """
    HashMap cache: date → StockRecord.
    Build O(n), lookup O(1), Tốt hơn O(n) của linear search.
    """
    def __init__(self, records: list):
        self._cache = {r.date: r for r in records}

    def get(self, d: date):
        return self._cache.get(d)

    def has_date(self, d: date) -> bool:
        return d in self._cache


# ---------- LAYER 3: PROCESSING — SLIDING WINDOW ----------
def sliding_window_ma(prices: list, k: int) -> list:
    """
    Moving Average bằng Sliding Window — O(n) time, O(k) space.
    Dùng dslk đôi deque để popleft() chỉ tốn O(1) thay vì mảng động list.pop(0) phải tốn O(n).
    idea leetcode 643
    """
    if len(prices) < k:
        return [None] * len(prices)
    result   = [None] * (k - 1)
    window   = deque()
    win_sum  = 0.0
    for price in prices:
        window.append(price)
        win_sum += price
        if len(window) > k:
            win_sum -= window.popleft()
        if len(window) == k:
            result.append(round(win_sum / k, 2))
    return result

def calc_ema(prices: list, k: int) -> list:
    """
    Exponential Moving Average — trọng số cao hơn cho giá gần đây.
    O(n) time, O(1) space. Dùng cho MACD.
    """
    if len(prices) < k:
        return [None] * len(prices)
    result    = [None] * (k - 1)
    mult      = 2 / (k + 1)
    ema       = sum(prices[:k]) / k
    result.append(round(ema, 2))
    
    for price in prices[k:]:
        ema = price * mult + ema * (1 - mult)
        result.append(round(ema, 2))
    return result

def calc_rsi(prices: list, period: int = 14) -> list:
    """
    Relative Strength Index — Sliding Window tính avg gain/loss.
    RSI > 70: overbought (có thể bán), RSI < 30: oversold (có thể mua).
    O(n) time, O(period) space.
    """
    result  = [None] * period
    gains   = deque(maxlen=period)
    losses  = deque(maxlen=period)
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))
        if len(gains) == period:
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss == 0:
                result.append(100.0)
            else:
                rs  = avg_gain / avg_loss
                result.append(round(100 - 100 / (1 + rs), 2))
    return result

def calc_bollinger_bands(prices: list, k: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands — Sliding Window tính mean ± std.
    Upper/Lower band xác định vùng giá bất thường.
    O(n) time, O(k) space.
    Middle = Trung bình cộng 20 ngày trước đó.
    Upper = Trung bình cộng 20 ngày trước đó + 2*(độ lệch chuẩn của 20 ngày trước đó).
    Lower = Trung bình cộng 20 ngày trước đó + 2*(độ lệch chuẩn của 20 ngày trước đó).
    """
    upper, middle, lower = [], [], []
    window = deque(maxlen=k)
    for price in prices:
        window.append(price)
        if len(window) < k:
            upper.append(None); middle.append(None); lower.append(None)
        else:
            avr  = sum(window) / k
            std = math.sqrt(sum((x - avr) ** 2 for x in window) / k)
            middle.append(round(avr, 2))
            upper.append(round(avr + num_std * std, 2))
            lower.append(round(avr - num_std * std, 2))
    return upper, middle, lower

# ---------- LAYER 4: ALGORITHM CORE ----------
def max_profit_v1(prices: list) -> tuple:
    """
    DP V1 — 1 giao dịch tối ưu.
    Source: leetcode 121
    Idea: duyệt qua danh sách giá một lần, giữ lại giá mua thấp nhất và tính lợi nhuận cao nhất có thể có tại mỗi thời điểm.
    O(n) time, O(1) space.
    Returns: profit, ngày mua, ngày bán
    """
    min_price = float('inf')
    min_idx   = 0
    max_profit = float('-inf')
    buy_idx = 0
    sell_idx = 0

    for i  in range(len(prices)):
        price = prices[i]
        if price < min_price:
            min_price = price
            min_idx   = i
        profit = price - min_price
        if profit > max_profit:
            max_profit = profit
            buy_idx    = min_idx
            sell_idx   = i
    return max_profit, buy_idx, sell_idx

def max_profit_v2(prices: list) -> tuple:
    """
    DP V2 — vô hạn giao dịch.
    Source: leetcode 122
    Idea: duyệt qua danh sách giá một lần, mô phỏn lướt sóng thị trường, gặp đáy là mua vào, gặp đỉnh là bán ra để tính tổng lãi cao nhất có thể đạt được.
    Insight: tổng lợi nhuận = tổng mọi đoạn tăng giá liên tiếp.
    O(n) time, O(1) space.
    Returns: (profit, list of (buy_idx, sell_idx))
    """
    profit = 0
    trades = []
    i = 0
    while i < len(prices) - 1:

        # Tìm đáy
        while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
            i += 1
        buy = i

        # Tìm đỉnh
        while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
            i += 1
        sell = i

        if sell > buy:
            profit += prices[sell] - prices[buy]
            trades.append((buy, sell))
    return profit, trades

# ---------- LAYER 5: SIGNAL GENERATION ----------
def generate_signals(prices, ma5, ma20, rsi):
    """
    Sinh tín hiệu Buy/Sell hiện chỉ dựa trên các tín hiệu từ đường MA và RSI, tín hiệu sinh ra từ các đường còn lại sẽ đc nâng cấp trong tương lai.
    Cách sinh tín hiệu Buy/Sell:
    - Golden Cross: MA5 cắt MA20 từ dưới lên ->  BUY
    - Death Cross:  MA5 cắt MA20 từ trên xuống -> SELL
    - RSI < 30 -> BUY confirmation
    - RSI > 70 -> SELL confirmation
    Chọn ưu tiên RSI vì RSI phản ánh sức mạnh xu hướng ngắn hạn. 
    Khi RSI quá mua/bán, thị trường có thể sớm đảo chiều, vì vậy luôn ưu tiên tín hiệu từ RSI tránh rủi ro.
    """
    signals = [None] * len(prices)
    for i in range(1, len(prices)):
        if ma5[i] is None or ma20[i] is None:
            continue
        if ma5[i - 1] is not None and ma20[i - 1] is not None:
            # Golden Cross
            if ma5[i - 1] <= ma20[i - 1] and ma5[i] > ma20[i]:
                signals[i] = 'BUY'
            # Death Cross
            elif ma5[i - 1] >= ma20[i - 1] and ma5[i] < ma20[i]:
                signals[i] = 'SELL'
        # RSI
        if rsi[i] is not None:
            if rsi[i] < 30 and signals[i] != 'BUY':
                signals[i] = 'BUY'
            elif rsi[i] > 70 and signals[i] != 'SELL':
                signals[i] = 'SELL'
    return signals


# ---------- LAYER 6: BACKTESTING ----------
def backtest(prices, signals, initial_capital=100000000):
    """
    Chạy backtest: mua khi BUY, bán khi SELL.
    Trả về portfolio value theo từng ngày để vẽ biểu đồ.
    Giả sử mô hình này mua cổ phiếu theo lô 100 để kiểm tra khả năng mua và tính toán hiệu suất đầu tư của mô hình.
    """
    capital   = initial_capital
    shares    = 0
    portfolio = []
    trades_log = []
    for i, (price, sig) in enumerate(zip(prices, signals)):
        if sig == 'BUY' and capital >= price:
            n_buy   = int(capital // price // 100) * 100  # mua theo lô 100
            if n_buy > 0:
                shares   += n_buy
                capital  -= n_buy * price
                trades_log.append({'day': i, 'type': 'BUY', 'price': price, 'shares': n_buy})
        elif sig == 'SELL' and shares > 0:
            capital += shares * price
            trades_log.append({'day': i, 'type': 'SELL', 'price': price, 'shares': shares})
            shares   = 0
        portfolio.append(capital + shares * price)
    final_value  = portfolio[-1] if portfolio else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100
    return portfolio, trades_log, total_return


# ---------- ENTRYPOINT ----------
def run_analysis(symbol="VNM"):
    print(f"\n{'='*55}")
    print(f"  STOCK ANALYSIS ENGINE — {symbol}")
    print(f"{'='*55}")
    # Bước 1: Thu thập dữ liệu
    records = fetch_data(symbol)
    store   = StockDataStore(records)
    cache   = StockCache(records)
    print(f"[DATA]  Loaded {len(records)} records | {records[0].date} → {records[-1].date}")
    prices  = store.prices


    # Bước 2: Tính chỉ số kỹ thuật
    ma5          = sliding_window_ma(prices, 5)
    ma20         = sliding_window_ma(prices, 20)
    ema12        = calc_ema(prices, 12)
    rsi          = calc_rsi(prices, 14)
    bb_up, bb_mid, bb_lo = calc_bollinger_bands(prices, 20)
    print(f"[PROC]  MA5, MA20, EMA12, RSI, Bollinger Bands — computed")


    # Bước 3: DP tối ưu lợi nhuận
    p1, bi, si   = max_profit_v1(prices)
    p2, trades2  = max_profit_v2(prices)
    print(f"V1 best profit: {p1:,.0f} (buy D{bi}→sell D{si})")
    print(f"V2 unlimited:   {p2:,.0f} ({len(trades2)} trades)")


    # Bước 4: Sinh tín hiệu & backtest
    signals      = generate_signals(prices, ma5, ma20, rsi)
    portfolio, trades_log, ret = backtest(prices, signals)
    n_buy  = sum(1 for t in trades_log if t['type'] == 'BUY')
    n_sell = sum(1 for t in trades_log if t['type'] == 'SELL')
    print(f"[SIG]   {n_buy} BUY signals, {n_sell} SELL signals")
    print(f"[BACK]  Total return: {ret:+.2f}%")
    print(f"{'='*55}\n")

    
    # Trả về tất cả dữ liệu để dashboard sử dụng
    return {
        "symbol":      symbol,
        "records":     records,
        "prices":      prices,
        "dates":       [str(r.date) for r in records],
        "highs":       store.highs,
        "lows":        store.lows,
        "volumes":     store.volumes,
        "ma5":         ma5,
        "ma20":        ma20,
        "ema12":       ema12,
        "rsi":         rsi,
        "bb_upper":    bb_up,
        "bb_middle":   bb_mid,
        "bb_lower":    bb_lo,
        "signals":     signals,
        "portfolio":   portfolio,
        "trades_log":  trades_log,
        "total_return": ret,
        "dp_v1":       {"profit": p1, "buy_idx": bi, "sell_idx": si},
        "dp_v2":       {"profit": p2, "trades": trades2},
    }

import matplotlib.pyplot as plt


def plot_results(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Mising matplotlib. Please run: pip install matplotlib")
        return

    prices = data['prices']
    dates = data['dates']
    ma5 = data['ma5']
    ma20 = data['ma20']
    signals = data['signals']

    plt.figure(figsize=(14, 7))
    
    # Plot prices and Moving Averages
    plt.plot(prices, label='Close Price', color='black', linewidth=1.5, alpha=0.7)
    plt.plot(ma5, label='MA 5', color='blue', linestyle='--', alpha=0.8)
    plt.plot(ma20, label='MA 20', color='orange', linestyle='--', alpha=0.8)

    # Plot Buy/Sell Markers
    buy_x = [i for i, s in enumerate(signals) if s == 'BUY']
    buy_y = [prices[i] for i in buy_x]
    
    sell_x = [i for i, s in enumerate(signals) if s == 'SELL']
    sell_y = [prices[i] for i in sell_x]

    plt.scatter(buy_x, buy_y, marker='^', color='green', s=120, label='BUY Signal', zorder=5)
    plt.scatter(sell_x, sell_y, marker='v', color='red', s=120, label='SELL Signal', zorder=5)

    # Add max profit V1 markers (optimal global trade)
    dp_v1 = data['dp_v1']
    if dp_v1['profit'] > 0:
        plt.scatter(dp_v1['buy_idx'], prices[dp_v1['buy_idx']], marker='o', color='lime', s=150, zorder=4, label=f"DP V1 Best Buy (D{dp_v1['buy_idx']})")
        plt.scatter(dp_v1['sell_idx'], prices[dp_v1['sell_idx']], marker='o', color='darkred', s=150, zorder=4, label=f"DP V1 Best Sell (D{dp_v1['sell_idx']})")

    # Format X-axis
    step = max(1, len(dates) // 10)
    plt.xticks(ticks=range(0, len(dates), step), labels=[dates[i] for i in range(0, len(dates), step)], rotation=45)
    
    plt.title(f"{data['symbol']} Stock Price & Trading Signals")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Welcome to Stock Technical Analysis and Trading Signal Generator - A DSA Project!")
    while True:
        symbol = input("\nEnter a stock symbol to analyze (or 'quit' to exit): ").strip().upper()
        if symbol.lower() == 'quit':
            print("Exiting...")
            break
        if not symbol:
            continue
        
        try:
            data = run_analysis(symbol)
            plot_results(data)
        except Exception as e:
            print(f"An error occurred: {e}")
