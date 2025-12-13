def print_trades_report(trades):
    if not trades:
        print("=== NO CLOSED TRADES ===")
        return

    print("=====================================")
    print("           TRADES REPORT             ")
    print("=====================================")

    for t in trades:
        print(f"TRADE #{t['id']}")
        print(f"TYPE: {t['type']}")
        print(f"ENTRY DATE: {t['entry_date']}")
        print(f"ENTRY PRICE: {t['entry_price']:.4f}")

        if t.get("entry_rsi") is not None:
            print(f"ENTRY RSI: {t['entry_rsi']:.2f}")

        if t.get("entry_macd") is not None:
            print(f"ENTRY MACD: {t['entry_macd']:.6f}")

        if t.get("entry_macd_signal") is not None:
            print(f"ENTRY MACD SIGNAL: {t['entry_macd_signal']:.6f}")

        print(f"REASON ENTRY: {t['reason_entry']}")

        print(f"EXIT DATE: {t['exit_date']}")
        print(f"EXIT PRICE: {t['exit_price']:.4f}")
        print(f"PROFIT: {t['profit']:.4f}")
        print(f"REASON EXIT: {t['reason_exit']}")
        print("-------------------------------------")