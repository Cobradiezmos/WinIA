import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPORTS


def kelly(p: float, odds: float) -> float:
    """Calcula la fracción teórica del criterio de Kelly para una apuesta."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def evaluate_bet(stake: float, odds: float, won: bool) -> float:
    """Calcula el rendimiento neto en moneda para una apuesta dada."""
    if stake <= 0:
        return 0.0
    return stake * (odds - 1.0) if won else -stake


def run_backtest(
    df: pd.DataFrame,
    min_edge: float = 0.025,
    min_ev: float = 0.02,
    bankroll: float = 1000.0,
    kelly_frac: float = 0.25,
    max_stake_pct: float = 0.02,
    max_daily_exposure_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ejecuta una simulación histórica de apuestas con reescalado proporcional de
    exposición diaria para evitar el sobre-apalancamiento.
    """
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    done = df.dropna(subset=['home_goals', 'away_goals', 'result']).copy()

    bets_list = []
    equity_curve = []
    current_bankroll = bankroll

    equity_curve.append({
        'date': done['date'].min(),
        'bankroll': current_bankroll,
        'bets': 0,
        'profit': 0.0,
    })

    # Procesar partidos agrupados por fecha
    for date_val, group in done.groupby('date'):
        daily_candidates = []

        for _, match in group.iterrows():
            for side, col in [('home', 'B365H'), ('draw', 'B365D'), ('away', 'B365A')]:
                p_col = f'p_model_{side}'
                if col not in match or p_col not in match:
                    continue

                odds = float(match[col]) if pd.notna(match[col]) else 0.0
                p_model = float(match[p_col]) if pd.notna(match[p_col]) else 0.0

                if odds <= 1.0 or p_model <= 0:
                    continue

                p_market = 1.0 / odds
                edge = p_model - p_market
                ev = (p_model * odds) - 1.0

                if edge >= min_edge and ev >= min_ev:
                    k_full = kelly(p_model, odds)
                    proposed_stake_pct = min(k_full * kelly_frac, max_stake_pct)

                    if proposed_stake_pct > 0:
                        daily_candidates.append({
                            'match_id': match['match_id'],
                            'date': date_val,
                            'season': match.get('season', None),
                            'league': match.get('league', None),
                            'home_team': match['home_team'],
                            'away_team': match['away_team'],
                            'market': side,
                            'odds': odds,
                            'p_model': p_model,
                            'p_market': p_market,
                            'edge': edge,
                            'ev': ev,
                            'proposed_stake_pct': proposed_stake_pct,
                            'won': match['result'] == side[0].upper(),
                        })

        if not daily_candidates:
            continue

        # Control de exposición diaria: escala si la suma de apuestas supera el límite
        total_proposed_pct = sum(c['proposed_stake_pct'] for c in daily_candidates)
        scale_factor = 1.0

        if total_proposed_pct > max_daily_exposure_pct:
            scale_factor = max_daily_exposure_pct / total_proposed_pct

        for cand in daily_candidates:
            final_stake_pct = cand['proposed_stake_pct'] * scale_factor
            stake = current_bankroll * final_stake_pct
            pnl = evaluate_bet(stake, cand['odds'], cand['won'])
            current_bankroll += pnl

            cand['stake_pct'] = final_stake_pct
            cand['stake'] = stake
            cand['pnl'] = pnl
            cand['bankroll_after'] = current_bankroll
            bets_list.append(cand)

        equity_curve.append({
            'date': date_val,
            'bankroll': current_bankroll,
            'bets': len(bets_list),
            'profit': current_bankroll - bankroll,
        })

    bets_df = pd.DataFrame(bets_list)
    equity_df = pd.DataFrame(equity_curve)

    # Cálculo de métricas
    total_bets = len(bets_df)
    total_staked = bets_df['stake'].sum() if total_bets > 0 else 0.0
    total_profit = bets_df['pnl'].sum() if total_bets > 0 else 0.0
    roi = (total_profit / total_staked) if total_staked > 0 else 0.0
    win_rate = float((bets_df['pnl'] > 0).mean()) if total_bets > 0 else 0.0

    peak = equity_df['bankroll'].cummax()
    drawdown = (peak - equity_df['bankroll']) / peak
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

    metrics = {
        'initial_bankroll': bankroll,
        'final_bankroll': current_bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'total_bets': total_bets,
        'total_staked': total_staked,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
    }

    return bets_df, equity_df, metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest de estrategia de apuestas de fútbol')
    parser.add_argument('--input', default='data/processed/predictions_ensemble.csv', help='Ruta a las predicciones')
    parser.add_argument('--output-bets', default='reports/backtest_bets.csv', help='Ruta para guardar apuestas ejecutadas')
    parser.add_argument('--output-equity', default='reports/backtest_equity.csv', help='Ruta para guardar la curva de banca')
    parser.add_argument('--output-metrics', default='reports/backtest_metrics.json', help='Ruta para guardar el resumen de métricas')
    parser.add_argument('--min-edge', type=float, default=0.025, help='Ventaja mínima sobre el mercado')
    parser.add_argument('--min-ev', type=float, default=0.02, help='Valor esperado mínimo')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Capital inicial')
    parser.add_argument('--kelly-frac', type=float, default=0.25, help='Fracción de Kelly')
    parser.add_argument('--max-stake', type=float, default=0.02, help='Apuesta máxima por partido (% de banca)')
    parser.add_argument('--max-daily-exposure-pct', type=float, default=0.15, help='Exposición máxima total diaria (% de banca)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    bets_df, equity_df, metrics = run_backtest(
        df=df,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        bankroll=args.bankroll,
        kelly_frac=args.kelly_frac,
        max_stake_pct=args.max_stake,
        max_daily_exposure_pct=args.max_daily_exposure_pct,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    bets_df.to_csv(args.output_bets, index=False)
    equity_df.to_csv(args.output_equity, index=False)

    Path(args.output_metrics).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()