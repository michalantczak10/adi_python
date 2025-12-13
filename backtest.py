import pandas as pd


def backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000,
    stop_loss_pct: float = 0.04,
    take_profit_pct: float = 0.10,
    trailing_pct: float = 0.03,
    export_csv: bool = False,
    export_prefix: str = "backtest",
):
    df = df.copy()

    capital = initial_capital
    position = 0
    entry_price = 0.0

    stop_loss_price = None
    take_profit_price = None
    trailing_stop = None

    trades = []
    open_trade = None

    equity_curve = []

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        price = row["Close"]
        signal = row["SIGNAL"]

        if signal == "BUY" and position == 0:
            position = 1
            entry_price = price

            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)
            trailing_stop = entry_price * (1 - trailing_pct)

            open_trade = {
                "id": len([t for t in trades if t["type"] == "LONG"]) + 1,
                "type": "LONG",
                "entry_date": date,
                "entry_price": entry_price,
                "entry_rsi": row.get("RSI_14"),
                "entry_macd": row.get("MACD"),
                "entry_macd_signal": row.get("MACD_SIGNAL"),
                "reason_entry": "SIGNAL",
            }

        if position == 1:
            new_trailing = price * (1 - trailing_pct)
            if new_trailing > trailing_stop:
                trailing_stop = new_trailing

        exit_reason = None

        if position == 1:
            if price <= stop_loss_price:
                exit_reason = "STOP_LOSS"
            elif price <= trailing_stop:
                exit_reason = "TRAILING_STOP"
            elif price >= take_profit_price:
                exit_reason = "TAKE_PROFIT"
            elif signal == "SELL":
                exit_reason = "SIGNAL"

        if position == 1 and exit_reason:
            profit = price - entry_price

            capital += profit

            trades.append({
                **open_trade,
                "exit_date": date,
                "exit_price": price,
                "profit": profit,
                "reason_exit": exit_reason,
            })

            position = 0
            entry_price = 0.0
            stop_loss_price = None
            take_profit_price = None
            trailing_stop = None
            open_trade = None

        if position == 1:
            equity_curve.append(capital + (price - entry_price))
        else:
            equity_curve.append(capital)

    df["EQUITY"] = equity_curve
    df["ROLLING_MAX"] = df["EQUITY"].cummax()
    df["DRAWDOWN"] = df["EQUITY"] - df["ROLLING_MAX"]

    trades_df = pd.DataFrame(trades)

    if export_csv:
        trades_df.to_csv(f"{export_prefix}_trades.csv", index=False)
        df.to_csv(f"{export_prefix}_equity.csv")

    return {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return": capital - initial_capital,
        "num_trades": len(trades),
        "win_rate": (trades_df["profit"] > 0).mean() * 100 if len(trades_df) else 0,
        "max_drawdown": df["DRAWDOWN"].min(),
        "trades": trades,
        "trades_df": trades_df,
        "df": df,
    }