import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1 jeśli następna świeca wyżej, 0 jeśli niżej/równo
    df["TARGET_UP"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def train_ml_model(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna()

    feature_cols = ["RSI_14", "MACD", "MACD_SIGNAL", "BB_LOW", "BB_HIGH", "Close"]
    X = df[feature_cols]
    y = df["TARGET_UP"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    df["ML_PRED_UP"] = model.predict(X)
    df["ML_PROBA_UP"] = model.predict_proba(X)[:, 1]

    return df, model