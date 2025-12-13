import yfinance as yf

from indicators import add_indicators
from ml_model import (
    add_target,
    add_target_pct,
    train_ml_model,
    ensemble_predict,
)
from backtest import backtest
from signals_solana import generate_signals_solana


class MLManager:
    def __init__(
        self,
        ticker: str,
        period: str,
        interval: str,
        initial_capital: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        trailing_pct: float,
    ):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_pct = trailing_pct

        self.data = None
        self.df_ml = None
        self.model = None
        self.metrics = None

    # =========================
    # DATA LOADING
    # =========================
    def load_data(self):
        print(f"Loading data: {self.ticker}, period={self.period}, interval={self.interval}")
        data = yf.download(self.ticker, period=self.period, interval=self.interval)
        if data is None or len(data) == 0:
            raise ValueError("No data downloaded")

        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data = add_indicators(data)
        self.data = data
        return self.data

    # =========================
    # BASIC ML TRAINING
    # =========================
    def train(self, horizon: int = 1, model_name: str = "rf"):
        if self.data is None:
            self.load_data()

        df_ml, model, metrics = train_ml_model(
            self.data,
            horizon=horizon,
            model_name=model_name,
        )

        self.df_ml = df_ml
        self.model = model
        self.metrics = metrics

        return df_ml, model, metrics

    # =========================
    # ML REPORT
    # =========================
    def report(self, ensemble: bool = False):
        print("=== ML REPORT ===")

        if self.data is None:
            self.load_data()

        if ensemble:
            print("Using ENSEMBLE")
            # Trenujemy osobne modele i robimy ensemble
            df_ml, _, base_metrics = self.train()
            X = df_ml[base_metrics["features"]]

            models = []
            for name in ["rf", "gb", "logreg"]:
                _, model, _ = train_ml_model(self.data, model_name=name)
                models.append(model)

            pred, proba = ensemble_predict(models, X)
            df_ml["ML_PRED_UP"] = pred
            df_ml["ML_PROBA_UP"] = proba

            print("\n=== SAMPLE ENSEMBLE PREDICTIONS (LAST 20) ===")
            print(df_ml[["Close", "ML_PRED_UP", "ML_PROBA_UP"]].tail(20))
            return df_ml, models, base_metrics

        df_ml, model, metrics = self.train()

        print("\n=== ML METRICS ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")

        print("\n=== CONFUSION MATRIX ===")
        cm = metrics["confusion_matrix"]
        print(f"TN={cm[0][0]}   FP={cm[0][1]}")
        print(f"FN={cm[1][0]}   TP={cm[1][1]}")

        if metrics.get("feature_importances"):
            print("\n=== FEATURE IMPORTANCE ===")
            for feat, imp in zip(metrics["features"], metrics["feature_importances"]):
                print(f"{feat:15s} -> {imp:.4f}")
        else:
            print("\n(No feature_importances for this model)")

        print("\n=== SAMPLE PREDICTIONS (LAST 20) ===")
        print(df_ml[["Close", "ML_PRED_UP", "ML_PROBA_UP"]].tail(20))

        return df_ml, model, metrics

    # =========================
    # ML SIGNAL
    # =========================
    def add_ml_signal(self, df, upper: float = 0.55, lower: float = 0.45):
        def classify(p: float) -> str:
            if p >= upper:
                return "BUY"
            if p <= lower:
                return "SELL"
            return "HOLD"

        df["ML_SIGNAL"] = df["ML_PROBA_UP"].apply(classify)
        return df

    # =========================
    # BACKTEST ML ONLY
    # =========================
    def backtest_ml_only(self):
        print("=== BACKTEST ML ONLY ===")
        df_ml, _, _ = self.train()
        df_ml = self.add_ml_signal(df_ml)

        print("\n=== ML SIGNAL DISTRIBUTION ===")
        print(df_ml["ML_SIGNAL"].value_counts())

        df_bt = df_ml.copy()
        df_bt["SIGNAL"] = df_bt["ML_SIGNAL"]

        results = backtest(
            df_bt,
            initial_capital=self.initial_capital,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_pct=self.trailing_pct,
        )

        print("\n=== BACKTEST RESULTS (ML ONLY) ===")
        for k, v in results.items():
            print(f"{k}: {v}")

        print("\n=== SAMPLE ML SIGNALS (LAST 15) ===")
        print(
            df_bt[
                [
                    "Close",
                    "RSI_14",
                    "MACD",
                    "MACD_SIGNAL",
                    "ML_PROBA_UP",
                    "ML_SIGNAL",
                ]
            ].tail(15)
        )

        return results

    # =========================
    # BACKTEST WITH ML FILTER
    # =========================
    def backtest_ml_filter(self):
        print("=== BACKTEST WITH ML FILTER ===")
        df_ml, _, _ = self.train()
        df = generate_signals_solana(df_ml)
        df = self.add_ml_signal(df)

        def combine(row):
            if row["SIGNAL"] == "BUY" and row["ML_SIGNAL"] == "BUY":
                return "BUY"
            if row["SIGNAL"] == "SELL" and row["ML_SIGNAL"] == "SELL":
                return "SELL"
            return "HOLD"

        df["FINAL_SIGNAL"] = df.apply(combine, axis=1)
        df["SIGNAL"] = df["FINAL_SIGNAL"]

        print("\n=== FINAL SIGNAL DISTRIBUTION ===")
        print(df["FINAL_SIGNAL"].value_counts())

        results = backtest(
            df,
            initial_capital=self.initial_capital,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_pct=self.trailing_pct,
        )

        print("\n=== BACKTEST RESULTS (WITH ML FILTER) ===")
        for k, v in results.items():
            print(f"{k}: {v}")

        return results

    # =========================
    # BACKTEST HYBRID (strategia + ML)
    # =========================
    def backtest_ml_hybrid(self):
        print("=== BACKTEST ML HYBRID ===")
        df_ml, _, _ = self.train()
        df = generate_signals_solana(df_ml)
        df = self.add_ml_signal(df)

        def hybrid(row):
            if row["SIGNAL"] == "BUY" and row["ML_SIGNAL"] == "BUY":
                return "BUY"
            if row["SIGNAL"] == "SELL" and row["ML_SIGNAL"] == "SELL":
                return "SELL"
            return "HOLD"

        df["FINAL_SIGNAL"] = df.apply(hybrid, axis=1)
        df["SIGNAL"] = df["FINAL_SIGNAL"]

        print("\n=== FINAL SIGNAL DISTRIBUTION (HYBRID) ===")
        print(df["FINAL_SIGNAL"].value_counts())

        results = backtest(
            df,
            initial_capital=self.initial_capital,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_pct=self.trailing_pct,
        )

        print("\n=== BACKTEST RESULTS (ML HYBRID) ===")
        for k, v in results.items():
            print(f"{k}: {v}")

        return results

    # =========================
    # OPTIMIZE ML THRESHOLDS
    # =========================
    def optimize_thresholds(self):
        print("=== OPTIMIZING ML THRESHOLDS ===")
        df_ml, _, _ = self.train()

        best_capital = -999999
        best_cfg = None

        for upper in [0.55, 0.60, 0.65, 0.70]:
            for lower in [0.45, 0.40, 0.35, 0.30]:

                df = self.add_ml_signal(df_ml.copy(), upper, lower)
                df["SIGNAL"] = df["ML_SIGNAL"]

                res = backtest(
                    df,
                    initial_capital=self.initial_capital,
                    stop_loss_pct=self.stop_loss_pct,
                    take_profit_pct=self.take_profit_pct,
                    trailing_pct=self.trailing_pct,
                )

                final_capital = res["final_capital"]

                print(f"upper={upper}, lower={lower} -> final={final_capital}")

                if final_capital > best_capital:
                    best_capital = final_capital
                    best_cfg = {"upper": upper, "lower": lower}

        print("\n=== BEST ML THRESHOLDS ===")
        print(best_cfg)
        print("Final capital:", best_capital)

        return best_cfg, best_capital

    # =========================
    # HORIZON SEARCH
    # =========================
    def search_horizon(self):
        print("=== ML HORIZON SEARCH ===")

        if self.data is None:
            self.load_data()

        best_f1 = None
        best_h = None

        for h in [1, 2, 4, 8]:
            df = add_target(self.data, horizon=h)
            _, _, metrics = train_ml_model(df, horizon=h)

            print(f"H={h} -> F1={metrics['f1']:.4f}")

            if best_f1 is None or metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_h = h

        print("\n=== BEST HORIZON ===")
        print("Horizon:", best_h, "F1:", best_f1)

        return best_h, best_f1

    # =========================
    # PERCENT MOVE MODEL
    # =========================
    def train_pct(self, horizon: int = 4, threshold: float = 0.02):
        print("=== ML PERCENT MOVE MODEL ===")

        if self.data is None:
            self.load_data()

        df = add_target_pct(self.data, horizon=horizon, threshold=threshold)
        df_ml, model, metrics = train_ml_model(df, horizon=horizon)

        print("\n=== METRICS (PCT MODEL) ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")

        print("\n=== SAMPLE PREDICTIONS (LAST 20) ===")
        print(df_ml[["Close", "ML_PROBA_UP", "ML_PRED_UP"]].tail(20))

        return df_ml, model, metrics
