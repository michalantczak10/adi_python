import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    rsi_buy: float = 30,
    rsi_sell: float = 70
) -> pd.DataFrame:
    df = df.copy()
    df["SIGNAL"] = "HOLD"

    for i in range(1, len(df)):
        rsi = df["RSI_14"].iloc[i]
        macd = df["MACD"].iloc[i]
        macd_signal = df["MACD_SIGNAL"].iloc[i]
        price = df["Close"].iloc[i]
        bb_low = df["BB_LOW"].iloc[i]
        bb_high = df["BB_HIGH"].iloc[i]

        # BUY: wyprzedanie + MACD w górę + dotknięcie dolnej wstęgi
        if (
            rsi < rsi_buy and
            macd > macd_signal and
            price <= bb_low
        ):
            df.at[df.index[i], "SIGNAL"] = "BUY"

        # SELL: wykupienie + MACD w dół + dotknięcie górnej wstęgi
        elif (
            rsi > rsi_sell and
            macd < macd_signal and
            price >= bb_high
        ):
            df.at[df.index[i], "SIGNAL"] = "SELL"

        else:
            df.at[df.index[i], "SIGNAL"] = "HOLD"

    return df