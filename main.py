import os
import warnings
import yfinance as yf
import matplotlib.pyplot as plt

from indicators import add_indicators
from signals_solana import generate_signals_solana
from backtest import backtest
from params import load_params
from ml_manager import MLManager

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['SSL_CERT_FILE'] = r"C:\certifi\cacert.pem"
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\certifi\cacert.pem"


# =========================
# KONFIGURACJA GŁÓWNA
# =========================

TICKER = "SOL-USD"
PERIOD = "max"
INTERVAL = "1h"

params = load_params()

INITIAL_CAPITAL = 10_000
STOP_LOSS = params.get("stop_loss", 0.04)
TAKE_PROFIT = params.get("take_profit", 0.12)
TRAILING_PCT = params.get("trailing", 0.04)

RUN_MODE = "BACKTEST"  # zmieniaj: BACKTEST, OPTIMIZE_SOL, ML_ONLY, BACKTEST_ML_ONLY, BACKTEST_WITH_ML_FILTER, BACKTEST_ML_HYBRID, OPTIMIZE_ML_THRESHOLDS, ML_HORIZON_SEARCH, ML_PCT, ML_ENSEMBLE

# =========================
# FUNKCJE BACKTEST / OPTIMIZE
# =========================

def run_backtest(return_results: bool = False):
    print("=== BACKTEST MODE ===")

    data = yf.download(TICKER, period=PERIOD, interval=INTERVAL)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data = add_indicators(data)
    df = generate_signals_solana(data)

    results = backtest(
        df,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=STOP_LOSS,
        take_profit_pct=TAKE_PROFIT,
        trailing_pct=TRAILING_PCT,
    )

    print("\n=== BACKTEST RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    if return_results:
        return results


def run_optimize_solana():
    from optimize_solana import optimize_solana

    optimize_solana()


# =========================
# GŁÓWNY ENTRYPOINT
# =========================

def main():
    manager = MLManager(
        ticker=TICKER,
        period=PERIOD,
        interval=INTERVAL,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=STOP_LOSS,
        take_profit_pct=TAKE_PROFIT,
        trailing_pct=TRAILING_PCT,
    )

    if RUN_MODE == "BACKTEST":
        run_backtest()

    elif RUN_MODE == "OPTIMIZE_SOL":
        run_optimize_solana()

    elif RUN_MODE == "ML_ONLY":
        manager.report()

    elif RUN_MODE == "ML_ENSEMBLE":
        manager.report(ensemble=True)

    elif RUN_MODE == "BACKTEST_ML_ONLY":
        manager.backtest_ml_only()

    elif RUN_MODE == "BACKTEST_WITH_ML_FILTER":
        manager.backtest_ml_filter()

    elif RUN_MODE == "BACKTEST_ML_HYBRID":
        manager.backtest_ml_hybrid()

    elif RUN_MODE == "OPTIMIZE_ML_THRESHOLDS":
        manager.optimize_thresholds()

    elif RUN_MODE == "ML_HORIZON_SEARCH":
        manager.search_horizon()

    elif RUN_MODE == "ML_PCT":
        manager.train_pct()

    else:
        print(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()
