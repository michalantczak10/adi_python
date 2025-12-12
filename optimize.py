from copy import deepcopy
from backtest import backtest
from indicators import add_indicators
from signals import generate_signals
import yfinance as yf


def simple_grid_search(ticker="AAPL", period="1y"):
    data = yf.download(ticker, period=period)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    best = None
    best_cfg = None

    for rsi_buy in [25, 30]:
        for rsi_sell in [70, 75]:
            for sl in [0.01, 0.02]:
                for tp in [0.03, 0.05]:
                    df = deepcopy(data)
                    df = add_indicators(df)
                    df = generate_signals(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)

                    res = backtest(
                        df,
                        initial_capital=10_000,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        enable_trailing=True,
                    )

                    final_capital = res["final_capital"]
                    if best is None or final_capital > best:
                        best = final_capital
                        best_cfg = {
                            "rsi_buy": rsi_buy,
                            "rsi_sell": rsi_sell,
                            "stop_loss": sl,
                            "take_profit": tp,
                        }

                    print(
                        f"CFG rsi_buy={rsi_buy}, rsi_sell={rsi_sell}, SL={sl}, TP={tp} -> final={final_capital}"
                    )

    print("\nBEST CONFIG:", best_cfg, "FINAL:", best)