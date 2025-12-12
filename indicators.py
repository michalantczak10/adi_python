import pandas as pd
import ta


def add_indicators(
    df: pd.DataFrame,
    rsi_window: int = 14,
    bb_window: int = 20,
    bb_dev: float = 2.0,
) -> pd.DataFrame:
    df = df.copy()

    # RSI
    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=rsi_window)
    df["RSI_14"] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close=df["Close"],
        window=bb_window,
        window_dev=bb_dev
    )
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_HIGH"] = bb.bollinger_hband()

    return df