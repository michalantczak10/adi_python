import os
os.environ['SSL_CERT_FILE'] = r"C:\certifi\cacert.pem"
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\certifi\cacert.pem"
import yfinance as yf
import matplotlib.pyplot as plt

from indicators import add_indicators
from signals import generate_signals
from backtest import backtest
from ml_model import add_target, train_ml_model


# --- 1. POBRANIE DANYCH ---

ticker = "AAPL"
period = "6mo"  # możesz zmienić na '1y', '2y', '5y', itd.

data = yf.download(ticker, period=period)

# spłaszczenie kolumn (na wypadek MultiIndex)
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]


# --- 2. WSKAŹNIKI ---

data = add_indicators(data)


# --- 3. SYGNAŁY ---

data = generate_signals(data)


# --- 4. BACKTEST ---

results = backtest(data)
bt_df = results["df"]

print("=== BACKTEST RESULTS ===")
print("Initial capital:", results["initial_capital"])
print("Final capital:", results["final_capital"])
print("Total return:", results["total_return"])
print("Num trades:", results["num_trades"])
print("Win rate:", f"{results['win_rate']:.2f}%")
print("Max drawdown:", results["max_drawdown"])
print()


# --- 5. ML: PRZYGOTOWANIE DANYCH + TRENING ---

bt_df = add_target(bt_df)
bt_df, model = train_ml_model(bt_df)


# --- 6. WIZUALIZACJA: CENA + BB + BUY/SELL ---

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: Cena + Bollinger + sygnały
ax1.plot(bt_df.index, bt_df["Close"], label="Close Price", color="blue")
ax1.plot(bt_df.index, bt_df["BB_LOW"], label="BB Low", color="green", linestyle="--")
ax1.plot(bt_df.index, bt_df["BB_HIGH"], label="BB High", color="red", linestyle="--")

buy_signals = bt_df[bt_df["SIGNAL"] == "BUY"]
sell_signals = bt_df[bt_df["SIGNAL"] == "SELL"]

ax1.scatter(buy_signals.index, buy_signals["Close"],
            marker="^", color="green", s=120, label="BUY")
ax1.scatter(sell_signals.index, sell_signals["Close"],
            marker="v", color="red", s=120, label="SELL")

ax1.set_title(f"{ticker} — Price with Bollinger Bands & Signals")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)

# Panel 2: RSI
ax2.plot(bt_df.index, bt_df["RSI_14"], label="RSI 14", color="purple")
ax2.axhline(30, color="green", linestyle="--", linewidth=1)
ax2.axhline(70, color="red", linestyle="--", linewidth=1)
ax2.set_title("RSI (14)")
ax2.set_ylabel("RSI")
ax2.legend()
ax2.grid(True)

# Panel 3: MACD
ax3.plot(bt_df.index, bt_df["MACD"], label="MACD", color="black")
ax3.plot(bt_df.index, bt_df["MACD_SIGNAL"], label="Signal", color="orange")
ax3.axhline(0, color="gray", linewidth=1)
ax3.set_title("MACD (12, 26, 9)")
ax3.set_ylabel("MACD")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()


# --- 7. WIZUALIZACJA: EQUITY CURVE ---

plt.figure(figsize=(12, 4))
plt.plot(bt_df.index, bt_df["EQUITY"], color="blue")
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Capital")
plt.grid(True)
plt.show()


# --- 8. ML: PODGLĄD PREDYKCJI ---

print("=== ML SAMPLE PREDICTIONS (last 10 rows) ===")
print(bt_df[["Close", "RSI_14", "MACD", "MACD_SIGNAL", "ML_PRED_UP", "ML_PROBA_UP"]].tail(10))