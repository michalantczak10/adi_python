import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Dodaje kolumnę TARGET_UP:
    1 - jeśli Close za `horizon` świec jest wyżej niż teraz
    0 - jeśli jest niżej lub równo
    """
    df = df.copy()
    df["TARGET_UP"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)
    return df


def _prepare_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Upewnia się, że target istnieje i że mamy komplet cech.
    """
    if "TARGET_UP" not in df.columns:
        df = add_target(df, horizon=horizon)

    df = df.dropna().copy()

    features = ["RSI_14", "MACD", "MACD_SIGNAL", "BB_LOW", "BB_HIGH", "Close"]
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Missing feature column: {f}")

    return df, features


def _build_model(model_name: str, random_state: int = 42):
    """
    Zwraca skonfigurowany model na podstawie nazwy.
    """
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    elif model_name == "logreg":
        return LogisticRegression(
            max_iter=500,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def ensemble_predict(models, X):
    probs = []
    for m in models:
        probs.append(m.predict_proba(X)[:, 1])
    avg = sum(probs) / len(probs)
    return (avg > 0.5).astype(int), avg

def add_target_pct(df, horizon=1, threshold=0.01):
    df = df.copy()
    future = df["Close"].shift(-horizon)
    pct = (future - df["Close"]) / df["Close"]
    df["TARGET_UP"] = (pct > threshold).astype(int)
    return df

def train_ml_model(
    df: pd.DataFrame,
    test_size: float = 0.3,
    horizon: int = 1,
    random_state: int = 42,
    model_name: str = "rf",
):
    """
    Trenuje wybrany model (rf / gb / logreg) do przewidywania TARGET_UP.

    Zwraca:
    - df z kolumnami ML_PRED_UP, ML_PROBA_UP na części testowej
    - wytrenowany model
    - słownik metryk
    """

    df, features = _prepare_features(df, horizon=horizon)

    if len(df) < 100:
        raise ValueError(f"Not enough data for ML training. Got {len(df)} rows.")

    X = df[features]
    y = df["TARGET_UP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = _build_model(model_name=model_name, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # LogisticRegression i GradientBoosting mają predict_proba, więc jest OK
    y_proba = model.predict_proba(X_test)[:, 1]

    df.loc[X_test.index, "ML_PRED_UP"] = y_pred
    df.loc[X_test.index, "ML_PROBA_UP"] = y_proba

    metrics = {
        "model_name": model_name,
        "horizon": horizon,
        "test_size": test_size,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "features": features,
    }

    # feature_importances tylko dla modeli, które je mają
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_.tolist()
    else:
        metrics["feature_importances"] = None

    return df, model, metrics