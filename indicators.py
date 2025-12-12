import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI 14
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

    # Bollinger Bands (20, 2)
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BB_LOW"] = bb["BBL_20_2.0"]
    df["BB_HIGH"] = bb["BBU_20_2.0"]

    return df
