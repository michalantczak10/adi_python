# =========================
# KONFIGURACJA
# =========================

TICKER = "AAPL"
PERIOD = "1y"
INITIAL_CAPITAL = 10_000
STOP_LOSS = 0.02
TAKE_PROFIT = 0.05
ML_THRESHOLD = 0.6

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# certyfikaty (jeśli potrzebujesz)
os.environ['SSL_CERT_FILE'] = r"C:\certifi\cacert.pem"
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\certifi\cacert.pem"

# =========================
# IMPORTY
# =========================

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

from indicators import add_indicators
from signals import generate_signals
from backtest import backtest
from ml_model import add_target, train_ml_model
# from optimize import simple_grid_search   # odpalasz tylko gdy chcesz


# =========================
# 1. POBRANIE DANYCH
# =========================

data = yf.download(TICKER, period=PERIOD)

# spłaszczenie MultiIndex
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]


# =========================
# 2. WSKAŹNIKI
# =========================

data = add_indicators(data)


# =========================
# 3. SYGNAŁY
# =========================

data = generate_signals(data)


# =========================
# 4. BACKTEST
# =========================

results = backtest(
    data,
    initial_capital=INITIAL_CAPITAL,
    stop_loss_pct=STOP_LOSS,
    take_profit_pct=TAKE_PROFIT,
    enable_trailing=True,
)

bt_df = results["df"]

print("=== BACKTEST RESULTS ===")
print("Initial capital:", results["initial_capital"])
print("Final capital:", results["final_capital"])
print("Total return:", results["total_return"])
print("Num trades:", results["num_trades"])
print("Win rate:", f"{results['win_rate']:.2f}%")
print("Max drawdown:", results["max_drawdown"])
print()


# =========================
# 5. ML – TRENING
# =========================

bt_df = add_target(bt_df)
bt_df, model = train_ml_model(bt_df)


# =========================
# 6. ML FILTR SYGNAŁÓW
# =========================

filtered = bt_df.copy()
filtered["SIGNAL_FILTERED"] = filtered["SIGNAL"]

for i in range(len(filtered)):
    if filtered["SIGNAL"].iloc[i] == "BUY":
        proba = filtered["ML_PROBA_UP"].iloc[i]
        if pd.isna(proba) or proba < ML_THRESHOLD:
            filtered.iat[i, filtered.columns.get_loc("SIGNAL_FILTERED")] = "HOLD"


# =========================
# 7. WIZUALIZACJA: CENA + BB + BUY/SELL
# =========================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel 1: Cena + Bollinger + sygnały
ax1.plot(bt_df.index, bt_df["Close"], label="Close Price", color="blue")
ax1.plot(bt_df.index, bt_df["BB_LOW"], label="BB Low", color="green", linestyle="--")
ax1.plot(bt_df.index, bt_df["BB_HIGH"], label="BB High", color="red", linestyle="--")

buy_signals = bt_df[bt_df["SIGNAL"] == "BUY"]
sell_signals = bt_df[bt_df["SIGNAL"] == "SELL"]

ax1.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", s=120)
ax1.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", s=120)

ax1.set_title(f"{TICKER} — Price with Bollinger Bands & Signals")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)

# Panel 2: RSI
ax2.plot(bt_df.index, bt_df["RSI_14"], label="RSI 14", color="purple")
ax2.axhline(30, color="green", linestyle="--")
ax2.axhline(70, color="red", linestyle="--")
ax2.set_title("RSI (14)")
ax2.set_ylabel("RSI")
ax2.legend()
ax2.grid(True)

# Panel 3: MACD
ax3.plot(bt_df.index, bt_df["MACD"], label="MACD", color="black")
ax3.plot(bt_df.index, bt_df["MACD_SIGNAL"], label="Signal", color="orange")
ax3.axhline(0, color="gray")
ax3.set_title("MACD (12, 26, 9)")
ax3.set_ylabel("MACD")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()


# =========================
# 8. EQUITY + DRAWDOWN
# =========================

fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(bt_df.index, bt_df["EQUITY"], color="blue", label="Equity")
ax1.set_ylabel("Equity", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(bt_df.index, bt_df["DRAWDOWN"], color="red", label="Drawdown")
ax2.set_ylabel("Drawdown", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Equity & Drawdown")
plt.grid(True)
plt.show()


# =========================
# 9. PODGLĄD ML
# =========================

print("=== ML SAMPLE PREDICTIONS (last 10 rows) ===")
print(bt_df[["Close", "RSI_14", "MACD", "MACD_SIGNAL", "ML_PRED_UP", "ML_PROBA_UP"]].tail(10))


# =========================
# 10. OPCJONALNIE: OPTIMALIZACJA
# =========================

# if __name__ == "__main__":
#     simple_grid_search(TICKER, PERIOD)