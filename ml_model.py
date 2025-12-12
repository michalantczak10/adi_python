import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TARGET_UP"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def train_ml_model(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna()

    feature_cols = ["RSI_14", "MACD", "MACD_SIGNAL", "BB_LOW", "BB_HIGH", "Close"]
    X = df[feature_cols]
    y = df["TARGET_UP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    df.loc[X_test.index, "ML_PRED_UP"] = model.predict(X_test)
    df.loc[X_test.index, "ML_PROBA_UP"] = model.predict_proba(X_test)[:, 1]

    return df, model