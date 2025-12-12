import pandas as pd


def backtest(df: pd.DataFrame, initial_capital: float = 10_000):
    df = df.copy()

    capital = initial_capital
    position = 0  # 0 = brak, 1 = long
    entry_price = 0.0
    trades = []
    equity_curve = []

    for i in range(len(df)):
        signal = df["SIGNAL"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        # BUY
        if signal == "BUY" and position == 0:
            position = 1
            entry_price = price
            trades.append(("BUY", date, price))

        # SELL
        elif signal == "SELL" and position == 1:
            profit = price - entry_price
            capital += profit
            position = 0
            trades.append(("SELL", date, price, profit))

        # equity
        if position == 1:
            equity_curve.append(capital + (price - entry_price))
        else:
            equity_curve.append(capital)

    df["EQUITY"] = equity_curve

    # statystyki
    total_return = capital - initial_capital
    closed_trades = [t for t in trades if len(t) == 4]
    wins = [t for t in closed_trades if t[3] > 0]
    losses = [t for t in closed_trades if t[3] <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    df["ROLLING_MAX"] = df["EQUITY"].cummax()
    df["DRAWDOWN"] = df["EQUITY"] - df["ROLLING_MAX"]
    max_drawdown = df["DRAWDOWN"].min() if len(df) > 0 else 0.0

    results = {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return": total_return,
        "num_trades": len(closed_trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "df": df,
    }

    return results