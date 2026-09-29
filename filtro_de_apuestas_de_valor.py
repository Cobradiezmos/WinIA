import argparse
import numpy as np
import pandas as pd


def kelly(p: float, odds: float) -> float:
    """Cálculo seguro del criterio de Kelly."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/processed/predictions_ensemble.csv')
    ap.add_argument('--output', default='data/processed/value_candidates.csv')
    ap.add_argument('--min-edge', type=float, default=0.025)
    ap.add_argument('--min-ev', type=float, default=0.02)
    ap.add_argument('--bankroll', type=float, default=1000.0)
    ap.add_argument('--kelly-frac', type=float, default=0.25)
    ap.add_argument('--max-stake-pct', type=float, default=0.02)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    rows = []

    trio_cols = {'home': 'B365H', 'draw': 'B365D', 'away': 'B365A'}
    has_full_trio = all(col in df.columns for col in trio_cols.values())

    # Calcular cuotas de mercado de-marginadas una sola vez si existen
    if has_full_trio:
        odds_trio = df[list(trio_cols.values())].apply(pd.to_numeric, errors='coerce')
        inv = 1.0 / odds_trio
        norm_df = inv.div(inv.sum(axis=1), axis=0)
    else:
        norm_df = None

    for side, oddcol in trio_cols.items():
        if oddcol not in df.columns:
            continue

        base_cols = ['date', 'league', 'season', 'home_team', 'away_team', 'match_id', f'p_model_{side}']
        x = df[base_cols].copy()
        x['odds'] = pd.to_numeric(df[oddcol], errors='coerce')
        
        # Filtrar cuotas inválidas y probabilidades fuera de rango
        valid_mask = (x['odds'] > 1.0) & x[f'p_model_{side}'].between(0.0001, 0.9999)
        x = x[valid_mask].copy()

        x['p_model'] = x[f'p_model_{side}']
        x['p_implied_raw'] = 1.0 / x['odds']

        # Ajuste de probabilidad de mercado usando índice explícito (previene desalineación)
        if has_full_trio and norm_df is not None:
            x['p_market_fair'] = norm_df.loc[x.index, trio_cols[side]]
        else:
            x['p_market_fair'] = x['p_implied_raw']

        # Métricas de valor y gestión de riesgo
        x['edge'] = x['p_model'] - x['p_market_fair']
        x['ev'] = (x['p_model'] * x['odds']) - 1.0
        x['kelly_full'] = [kelly(p, q) for p, q in zip(x['p_model'], x['odds'])]
        x['stake_pct'] = (x['kelly_full'] * args.kelly_frac).clip(0.0, args.max_stake_pct)
        x['stake'] = args.bankroll * x['stake_pct']
        x['market'] = side

        # Filtrar por requisitos mínimos de ventaja y EV
        filtered = x[(x['edge'] >= args.min_edge) & (x['ev'] >= args.min_ev)]
        rows.append(filtered)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(['date', 'ev'], ascending=[True, False])

    result.to_csv(args.output, index=False)
    print(f'Oportunidades de valor detectadas: {len(result)}')


if __name__ == '__main__':
    main()