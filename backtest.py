import pandas as pd


def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05,
    enable_trailing: bool = True,
):
    df = df.copy()

    capital = initial_capital
    position = 0  # 0 = brak, 1 = long
    entry_price = 0.0
    stop_loss_price = None
    take_profit_price = None
    trades = []
    equity_curve = []

    for i in range(len(df)):
        signal = df["SIGNAL"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        # otwarcie pozycji
        if signal == "BUY" and position == 0:
            position = 1
            entry_price = price
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
            trades.append(("BUY", date, price))

        # trailing stop - przesuwamy SL w górę, jeśli cena rośnie
        if position == 1 and enable_trailing:
            new_sl = price * (1 - stop_loss_pct)
            if new_sl > stop_loss_price:
                stop_loss_price = new_sl

        # warunki zamknięcia pozycji:
        exit_reason = None

        # 1) sygnał SELL
        if signal == "SELL" and position == 1:
            exit_reason = "SIGNAL"

        # 2) Stop Loss
        if position == 1 and price <= stop_loss_price:
            exit_reason = "STOP_LOSS"

        # 3) Take Profit
        if position == 1 and price >= take_profit_price:
            exit_reason = exit_reason or "TAKE_PROFIT"

        if position == 1 and exit_reason is not None:
            profit = price - entry_price
            capital += profit
            trades.append(("SELL", date, price, profit, exit_reason))
            position = 0
            entry_price = 0.0
            stop_loss_price = None
            take_profit_price = None

        # equity
        if position == 1:
            equity_curve.append(capital + (price - entry_price))
        else:
            equity_curve.append(capital)

    df["EQUITY"] = equity_curve

    closed_trades = [t for t in trades if len(t) >= 4]
    total_return = capital - initial_capital
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