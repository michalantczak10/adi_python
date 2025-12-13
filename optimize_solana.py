import yfinance as yf
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from indicators import add_indicators
from signals_solana import generate_signals_solana
from backtest import backtest
from params import save_params


def _evaluate_config(data, cfg, split_date=None):
    """
    Pomocnicza funkcja do oceny jednej konfiguracji:
    - data: DataFrame z danymi + wskaźnikami
    - cfg: dict z parametrami strategii
    - split_date: jeśli podane, dzielimy dane na in-sample / out-of-sample
    """

    df = deepcopy(data)

    df = generate_signals_solana(
        df,
        rsi_buy=cfg["rsi_buy"],
        rsi_sell=cfg["rsi_sell"],
        use_bollinger_filter=False,  # stabilniejsze do optymalizacji
    )

    # jeśli nie ma żadnego BUY -> bez sensu oceniać taką konfigurację
    if df["SIGNAL"].value_counts().get("BUY", 0) == 0:
        return None

    if split_date is not None:
        in_sample = df[df.index < split_date]
        out_sample = df[df.index >= split_date]

        # za mało danych -> pomijamy
        if len(in_sample) < 100 or len(out_sample) < 50:
            return None

        res_in = backtest(
            in_sample,
            initial_capital=10_000,
            stop_loss_pct=cfg["stop_loss"],
            take_profit_pct=cfg["take_profit"],
            trailing_pct=cfg["trailing"],
        )

        res_out = backtest(
            out_sample,
            initial_capital=res_in["final_capital"],
            stop_loss_pct=cfg["stop_loss"],
            take_profit_pct=cfg["take_profit"],
            trailing_pct=cfg["trailing"],
        )

        return {
            "cfg": cfg,
            "final_in": res_in["final_capital"],
            "final_out": res_out["final_capital"],
        }

    # BEZ out-of-sample – pełne dane
    res = backtest(
        df,
        initial_capital=10_000,
        stop_loss_pct=cfg["stop_loss"],
        take_profit_pct=cfg["take_profit"],
        trailing_pct=cfg["trailing"],
    )

    return {
        "cfg": cfg,
        "final_in": res["final_capital"],
        "final_out": res["final_capital"],
    }


def optimize_solana(
    ticker="SOL-USD",
    period="2y",
    interval="4h",
    use_out_of_sample=True,
    oos_fraction=0.25,  # ostatnie 25% danych jako out-of-sample
    max_workers=8,      # równoległość
):
    print(f"=== RUNNING OPTIMIZATION FOR {ticker} ===")
    print(f"Period: {period}, Interval: {interval}")
    print(f"Out-of-sample: {use_out_of_sample}, OOS fraction: {oos_fraction}\n")

    # --- 1. Pobranie danych ---
    data = yf.download(ticker, interval=interval, period=period)

    if data is None or len(data) < 300:
        print("ERROR: Too little data to optimize. Try different period or interval.")
        return

    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # --- 2. Dodanie wskaźników tylko raz ---
    data = add_indicators(data)

    # --- 3. Wyznaczenie daty podziału in-sample / out-of-sample ---
    split_date = None
    if use_out_of_sample:
        split_index = int(len(data) * (1 - oos_fraction))
        split_date = data.index[split_index]
        print(f"Split date (in-sample <, out-of-sample >=): {split_date}\n")

    # --- 4. Parametry do testowania ---
    rsi_buy_list = [35, 40, 45, 50]
    rsi_sell_list = [55, 60, 65]
    sl_list = [0.03, 0.04, 0.05]
    tp_list = [0.06, 0.08, 0.10, 0.12]
    trailing_list = [0.02, 0.03, 0.04]

    configs = []
    for rsi_buy in rsi_buy_list:
        for rsi_sell in rsi_sell_list:
            for sl in sl_list:
                for tp in tp_list:
                    for trailing in trailing_list:
                        configs.append({
                            "rsi_buy": rsi_buy,
                            "rsi_sell": rsi_sell,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "trailing": trailing,
                        })

    total_configs = len(configs)
    print(f"Total configs to evaluate: {total_configs}\n")

    best_cfg = None
    best_out = -999999
    best_in = -999999

    # --- 5. Równoległa ocena konfiguracji z prostym progress barem ---
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_evaluate_config, data, cfg, split_date)
            for cfg in configs
        ]

        for fut in as_completed(futures):
            result = fut.result()
            done += 1

            # PROGRESS
            progress = done / total_configs * 100
            print(f"\rProgress: {done}/{total_configs} ({progress:5.1f}%)", end="")

            if result is None:
                continue

            cfg = result["cfg"]
            final_in = result["final_in"]
            final_out = result["final_out"]

            # prosty log dla konfiguracji, które przeszły
            print(
                f"\nCFG rsi_buy={cfg['rsi_buy']}, rsi_sell={cfg['rsi_sell']}, "
                f"SL={cfg['stop_loss']}, TP={cfg['take_profit']}, TR={cfg['trailing']} "
                f"-> in={final_in:.2f}, out={final_out:.2f}"
            )

            # kryterium: maksymalizujemy wynik out-of-sample
            if final_out > best_out:
                best_out = final_out
                best_in = final_in
                best_cfg = cfg

    print("\n\n=== BEST CONFIG FOUND ✅ ===")
    if best_cfg is None:
        print("No valid configuration found.")
        return

    print("Best config:", best_cfg)
    print(f"In-sample final capital:  {best_in:.2f}")
    print(f"Out-of-sample final capital: {best_out:.2f}")

    # Zapis parametrów do JSON
    save_params(best_cfg)
    print("\nSaved to params.json ✅")

    return best_cfg