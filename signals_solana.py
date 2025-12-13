import pandas as pd


def generate_signals_solana(
    df,
    rsi_buy=45,
    rsi_sell=55,
    sma_window=200,
    use_bollinger_filter=True
) -> pd.DataFrame:

    df = df.copy()
    df["SIGNAL"] = "HOLD"

    df["SMA200"] = df["Close"].rolling(sma_window).mean()

    for i in range(1, len(df)):
        rsi = df["RSI_14"].iloc[i]
        macd = df["MACD"].iloc[i]
        macd_signal = df["MACD_SIGNAL"].iloc[i]
        price = df["Close"].iloc[i]
        sma200 = df["SMA200"].iloc[i]
        bb_low = df["BB_LOW"].iloc[i]
        bb_high = df["BB_HIGH"].iloc[i]

        buy_trend = (
            price > sma200 and
            macd > macd_signal and
            rsi < rsi_buy
        )

        buy_reversion = (
            rsi < rsi_buy and
            macd > macd_signal
        )

        if use_bollinger_filter:
            buy_reversion = buy_reversion and price <= bb_low

        if buy_trend or buy_reversion:
            df.at[df.index[i], "SIGNAL"] = "BUY"
            continue

        sell_condition = (
            rsi > rsi_sell and
            macd < macd_signal
        )

        if use_bollinger_filter:
            sell_condition = sell_condition and price >= bb_high

        if sell_condition:
            df.at[df.index[i], "SIGNAL"] = "SELL"
            continue

    return df