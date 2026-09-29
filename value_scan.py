import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def scan_value_bets(
    df, min_edge=0.025, min_ev=0.02, kelly_fraction=0.25, max_stake=0.02
):
    candidates = []

    # Definición de mapeo de mercados a escanear en Football-Data / Bet365
    # Estructura: (Nombre Mercado, Selección, Columna Probabilidad Modelo, Columnas de Cuota a comprobar)
    markets = [
        # Mercado 1X2
        ("1X2", "Home", "p_model_home", ["B365H", "BbAvH", "PSH", "MaxH"]),
        ("1X2", "Draw", "p_model_draw", ["B365D", "BbAvD", "PSD", "MaxD"]),
        ("1X2", "Away", "p_model_away", ["B365A", "BbAvA", "PSA", "MaxA"]),
        # Mercado Over / Under 2.5
        (
            "Over/Under 2.5",
            "Over 2.5",
            "p_over25",
            ["B365>2.5", "BbAv>2.5", "P>2.5", "Max>2.5"],
        ),
        (
            "Over/Under 2.5",
            "Under 2.5",
            "p_under25",
            ["B365<2.5", "BbAv<2.5", "P<2.5", "Max<2.5"],
        ),
        # Mercado Ambos Marcan (BTTS)
        (
            "BTTS",
            "BTTS Yes",
            "p_btts_yes",
            ["B365BTTSH", "BbAvBTTSH", "BTTS_Y"],
        ),
        (
            "BTTS",
            "BTTS No",
            "p_btts_no",
            ["B365BTTSA", "BbAvBTTSA", "BTTS_N"],
        ),
        # Mercado Handicap Asiatico 0 / DNB
        ("Asian Handicap 0", "AH 0 Home", "p_ah0_home", ["B365AHH", "BbAvAHH"]),
        ("Asian Handicap 0", "AH 0 Away", "p_ah0_away", ["B365AHA", "BbAvAHA"]),
    ]

    for idx, row in df.iterrows():
        for m_name, side, prob_col, odd_cols in markets:
            if prob_col not in row or pd.isna(row[prob_col]):
                continue

            prob = float(row[prob_col])

            # Buscar la mejor cuota disponible entre las columnas especificadas
            odd = None
            for col in odd_cols:
                if col in row and pd.notna(row[col]) and float(row[col]) > 1.0:
                    odd = float(row[col])
                    break

            if odd is None:
                continue

            # Cálculo de ventaja (Edge) y Expected Value (EV)
            edge = (prob * odd) - 1.0
            ev = edge  # EV por unidad apostada

            if edge >= min_edge and ev >= min_ev:
                # Criterio de Kelly
                b = odd - 1.0
                q = 1.0 - prob
                kelly_full = (b * prob - q) / b if b > 0 else 0
                kelly_full = max(0.0, kelly_full)

                # Stake recomendado (Kelly fraccionado con límite máximo)
                stake = min(kelly_full * kelly_fraction, max_stake)

                candidates.append({
                    "date": row["date"],
                    "league": row.get("league", row.get("Div", "N/A")),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "market": m_name,
                    "side": side,
                    "p_model": round(prob, 4),
                    "odd": odd,
                    "edge": round(edge, 4),
                    "ev": round(ev, 4),
                    "kelly": round(kelly_full, 4),
                    "stake": round(stake, 4),
                })

    return pd.DataFrame(candidates)


def main():
    parser = argparse.ArgumentParser(description="Escáner de valor WinIA")
    parser.add_argument(
        "--input", default="data/processed/predictions_ensemble.csv"
    )
    parser.add_argument(
        "--output", default="data/processed/value_candidates.csv"
    )
    parser.add_argument("--min-edge", type=float, default=0.025)
    parser.add_argument("--min-ev", type=float, default=0.02)
    args = parser.parse_args()

    input_path = Path(args.input)
    # Si no existe predictions_ensemble.csv, probar con predictions_elo_poisson.csv
    if not input_path.exists():
        fallback = Path("data/processed/predictions_elo_poisson.csv")
        if fallback.exists():
            input_path = fallback

    print(f"Cargando predicciones desde {input_path}...")
    df = pd.read_csv(input_path)

    df_val = scan_value_bets(
        df, min_edge=args.min_edge, min_ev=args.min_ev
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_val.to_csv(out_path, index=False)

    print(
        f"✓ Oportunidades de valor encontradas: {len(df_val)}. Guardado en {out_path}"
    )


if __name__ == "__main__":
    main()
