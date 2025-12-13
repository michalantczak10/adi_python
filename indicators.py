import pandas as pd
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_HIGH"] = bb.bollinger_hband()

    return df