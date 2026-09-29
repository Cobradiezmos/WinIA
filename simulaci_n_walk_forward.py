import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPORTS
from model_pipeline import (
    align_classifier_probs,
    build_classifier,
    build_logistic,
    fit_poisson_goals,
    fit_temperature,
    make_poisson_probs_vectorized,
    market_probs,
    normalize_probs,
    optimize_weights,
    predict_lambdas,
)


def kelly(p: float, odds: float) -> float:
    """Calcula el criterio de Kelly para una apuesta."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def evaluate_bet(stake: float, odds: float, won: bool) -> float:
    """Calcula el retorno neto de la apuesta."""
    if stake <= 0:
        return 0.0
    return stake * (odds - 1.0) if won else -stake


def run_walk_forward(
    df: pd.DataFrame,
    start_season: int = 2021,
    min_train: int = 400,
    min_edge: float = 0.025,
    min_ev: float = 0.02,
    bankroll: float = 1000.0,
    kelly_frac: float = 0.25,
    max_stake_pct: float = 0.02,
    max_daily_exposure_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simula una estrategia walk-forward con control de exposición diaria y
    gestión de riesgo de banca.
    """
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    done = df.dropna(subset=['home_goals', 'away_goals', 'result']).copy()

    seasons = [s for s in sorted(done.season.unique()) if s >= start_season]
    bets_list = []
    equity_curve = []

    current_bankroll = bankroll
    equity_curve.append({
        'date': done['date'].min(),
        'season': seasons[0] if seasons else None,
        'bankroll': current_bankroll,
        'bets': 0,
        'profit': 0.0,
    })

    for s in seasons:
        train = done[done.season < s]
        test = done[done.season == s]

        if len(train) < min_train or len(test) == 0:
            continue

        # Entrenamiento de modelos para la temporada actual
        pois = fit_poisson_goals(train)
        gb, gbcols = build_classifier(train)
        lr, lcols = build_logistic(train)

        # Usar los últimos partidos del conjunto de entrenamiento como calibración
        n_cal = min(len(train) // 4, 300)
        cal = train.iloc[-n_cal:]

        # Generar predicciones para la fase de calibración
        lh_c, la_c = predict_lambdas(pois, cal)
        pp_c = make_poisson_probs_vectorized(lh_c, la_c)
        pg_c = align_classifier_probs(gb, cal[gbcols])
        pl_c = align_classifier_probs(lr, cal[lcols])
        pm_c = market_probs(cal)
        pe_c = normalize_probs(np.column_stack([
            cal.elo_prob_home.values,
            (1 - cal.elo_prob_home.values) * 0.28,
            (1 - cal.elo_prob_home.values) * 0.72,
        ]))

        weights, _ = optimize_weights([pp_c, pg_c, pl_c, pe_c, pm_c], cal.result.values)
        pcal = sum(weights[i] * plist for i, plist in enumerate([pp_c, pg_c, pl_c, pe_c, pm_c]))
        temp = fit_temperature(pcal, cal.result.values)

        # Generar predicciones para el conjunto de prueba
        lh_t, la_t = predict_lambdas(pois, test)
        pp_t = make_poisson_probs_vectorized(lh_t, la_t)
        pg_t = align_classifier_probs(gb, test[gbcols])
        pl_t = align_classifier_probs(lr, test[lcols])
        pm_t = market_probs(test)
        pe_t = normalize_probs(np.column_stack([
            test.elo_prob_home.values,
            (1 - test.elo_prob_home.values) * 0.28,
            (1 - test.elo_prob_home.values) * 0.72,
        ]))

        probs_test = sum(weights[i] * plist for i, plist in enumerate([pp_t, pg_t, pl_t, pe_t, pm_t]))
        
        # Aplicar calibración por temperatura de forma estable
        logits = np.log(np.clip(probs_test, 1e-8, 1.0))
        exp_logits = np.exp(logits / temp)
        probs_calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        test_preds = test.copy()
        test_preds[['p_model_home', 'p_model_draw', 'p_model_away']] = probs_calibrated

        # Procesar apuestas agrupadas por fecha para limitar la exposición diaria
        for date_val, group in test_preds.groupby('date'):
            daily_candidates = []

            for _, match in group.iterrows():
                for side, col in [('home', 'B365H'), ('draw', 'B365D'), ('away', 'B365A')]:
                    if col not in match or pd.isna(match[col]):
                        continue

                    odds = float(match[col])
                    if odds <= 1.0:
                        continue

                    p_model = match[f'p_model_{side}']
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
                                'season': s,
                                'league': match['league'],
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

            # Ajustar apuestas si la exposición diaria supera el máximo permitido
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
                'season': s,
                'bankroll': current_bankroll,
                'bets': len(bets_list),
                'profit': current_bankroll - bankroll,
            })

    bets_df = pd.DataFrame(bets_list)
    equity_df = pd.DataFrame(equity_curve)

    # Cálculo de métricas agregadas
    total_bets = len(bets_df)
    total_staked = bets_df['stake'].sum() if total_bets > 0 else 0.0
    total_profit = bets_df['pnl'].sum() if total_bets > 0 else 0.0
    roi = (total_profit / total_staked) if total_staked > 0 else 0.0
    win_rate = (bets_df['pnl'] > 0).mean() if total_bets > 0 else 0.0

    peak = equity_df['bankroll'].cummax()
    drawdown = (peak - equity_df['bankroll']) / peak
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

    metrics = {
        'initial_bankroll': bankroll,
        'final_bankroll': current_bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'total_bets': total_bets,
        'win_rate': float(win_rate),
        'max_drawdown': max_drawdown,
    }

    return bets_df, equity_df, metrics


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Backtesting Framework')
    parser.add_argument('--input', default='data/processed/asof_features.csv', help='Ruta a las features as-of')
    parser.add_argument('--output-bets', default='reports/walk_forward_bets.csv', help='Ruta para guardar las apuestas')
    parser.add_argument('--output-equity', default='reports/walk_forward_equity.csv', help='Ruta para guardar la curva de capital')
    parser.add_argument('--output-metrics', default='reports/walk_forward_metrics.json', help='Ruta para guardar métricas')
    parser.add_argument('--start-season', type=int, default=2021, help='Temporada inicial de prueba')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Capital inicial')
    parser.add_argument('--kelly-frac', type=float, default=0.25, help='Fracción de Kelly')
    parser.add_argument('--max-stake', type=float, default=0.02, help='Apuesta máxima por partido (% de banca)')
    parser.add_argument('--max-daily-exposure', type=float, default=0.15, help='Exposición máxima diaria (% de banca)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=['date'])
    bets_df, equity_df, metrics = run_walk_forward(
        df=df,
        start_season=args.start_season,
        bankroll=args.bankroll,
        kelly_frac=args.kelly_frac,
        max_stake_pct=args.max_stake,
        max_daily_exposure_pct=args.max_daily_exposure,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    bets_df.to_csv(args.output_bets, index=False)
    equity_df.to_csv(args.output_equity, index=False)
    
    Path(args.output_metrics).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()