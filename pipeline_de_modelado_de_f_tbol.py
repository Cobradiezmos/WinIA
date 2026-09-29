import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FEATURES, MIN_MATCHES, MODELS, SEED

warnings.filterwarnings('ignore')

TARGETS = ['H', 'D', 'A']


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Calcula probabilidades implícitas del mercado priorizando Pinnacle, Bet365 o Promedios."""
    for trip in [('PSH', 'PSD', 'PSA'), ('B365H', 'B365D', 'B365A'), ('AvgH', 'AvgD', 'AvgA')]:
        if all(c in df.columns for c in trip):
            odds = df[list(trip)].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            inv = 1.0 / odds.where(odds > 1)
            sum_inv = inv.sum(axis=1)
            return inv.div(sum_inv, axis=0).fillna(1/3).to_numpy()
    return np.full((len(df), 3), 1/3)


def make_poisson_probs_vectorized(lh: np.ndarray, la: np.ndarray, rho: float = -0.05, max_goals: int = 10) -> np.ndarray:
    """
    Calcula probabilidades 1X2 usando Poisson bivariado con ajuste Dixon-Coles vectorizado.
    """
    gh = np.arange(max_goals + 1)
    ga = np.arange(max_goals + 1)
    
    # PMFs individuales: dimensiones (N, max_goals+1, 1) y (N, 1, max_goals+1)
    pmf_h = poisson.pmf(gh, lh[:, None, None])
    pmf_a = poisson.pmf(ga, la[:, None, None]).transpose(0, 2, 1)
    
    # Matriz bivariada independiente (N, max_goals+1, max_goals+1)
    P = pmf_h * pmf_a
    
    # Corrección Dixon-Coles para marcadores bajos
    tau_00 = np.maximum(0.01, 1 - lh * la * rho)
    tau_01 = np.maximum(0.01, 1 + lh * rho)
    tau_10 = np.maximum(0.01, 1 + la * rho)
    tau_11 = np.maximum(0.01, 1 - rho)
    
    P[:, 0, 0] *= tau_00
    P[:, 0, 1] *= tau_01
    P[:, 1, 0] *= tau_10
    P[:, 1, 1] *= tau_11
    
    # Normalizar suma
    sums = P.sum(axis=(1, 2), keepdims=True)
    P /= np.maximum(sums, 1e-12)
    
    # Extraer probabilidades 1X2
    p_home = np.tril(P, -1).sum(axis=(1, 2))
    p_draw = np.trace(P, axis1=1, axis2=2)
    p_away = np.triu(P, 1).sum(axis=(1, 2))
    
    probs = np.column_stack([p_home, p_draw, p_away])
    return probs / probs.sum(axis=1, keepdims=True)


def fit_poisson_goals(train: pd.DataFrame):
    num = [c for c in FEATURES if c in train.columns]
    cats = ['league', 'home_team', 'away_team']
    
    pre_h = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num)
    ], remainder='drop')
    
    pre_a = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num)
    ], remainder='drop')
    
    mh = Pipeline([('pre', pre_h), ('reg', PoissonRegressor(alpha=0.35, max_iter=500))])
    ma = Pipeline([('pre', pre_a), ('reg', PoissonRegressor(alpha=0.35, max_iter=500))])
    
    X = train[cats + num]
    mh.fit(X, train.home_goals.clip(lower=0))
    ma.fit(X, train.away_goals.clip(lower=0))
    return mh, ma


def predict_lambdas(models, df: pd.DataFrame):
    mh, ma = models
    X = df[['league', 'home_team', 'away_team'] + [c for c in FEATURES if c in df.columns]]
    return np.clip(mh.predict(X), 0.05, 6.0), np.clip(ma.predict(X), 0.05, 6.0)


def build_classifier(train: pd.DataFrame):
    cols = [c for c in FEATURES if c in train.columns]
    X = train[cols].replace([np.inf, -np.inf], np.nan)
    y = train.result
    
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('model', HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.035, max_leaf_nodes=15, 
            l2_regularization=2.0, random_state=SEED
        ))
    ])
    pipe.fit(X, y)
    return pipe, cols


def build_logistic(train: pd.DataFrame):
    cols = [c for c in FEATURES if c in train.columns]
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median', add_indicator=True)),
        ('sc', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000, C=0.4, random_state=SEED))
    ])
    pipe.fit(train[cols], train.result)
    return pipe, cols


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def align_classifier_probs(pipe, df_cols, targets=TARGETS) -> np.ndarray:
    q = pipe.predict_proba(df_cols)
    out = np.zeros((len(df_cols), 3))
    for j, c in enumerate(pipe.classes_):
        if c in targets:
            out[:, targets.index(c)] = q[:, j]
    return normalize_probs(out)


def optimize_weights(probs_list, y):
    Y = np.array([{'H': 0, 'D': 1, 'A': 2}[v] for v in y])
    k = len(probs_list)
    
    def obj(z):
        w = softmax(z)
        p = sum(w[i] * probs_list[i] for i in range(k))
        return log_loss(Y, p, labels=[0, 1, 2])
        
    res = minimize(obj, np.zeros(k), method='BFGS')
    return softmax(res.x), res.fun


def fit_temperature(p, y):
    Y = np.array([{'H': 0, 'D': 1, 'A': 2}[v] for v in y])
    logits = np.log(np.clip(p, 1e-8, 1.0))
    
    def loss(logt):
        return log_loss(Y, softmax(logits / np.exp(logt), axis=1), labels=[0, 1, 2])
        
    r = minimize_scalar(loss, bounds=(-2, 2), method='bounded')
    return float(np.exp(r.x))


def evaluate(p, y):
    Y = np.array([{'H': 0, 'D': 1, 'A': 2}[v] for v in y])
    return {
        'logloss': float(log_loss(Y, p, labels=[0, 1, 2])),
        'brier_mean': float(np.mean([brier_score_loss((Y == i).astype(int), p[:, i]) for i in range(3)])),
        'accuracy': float((p.argmax(1) == Y).mean())
    }


def train_and_predict(df: pd.DataFrame):
    done = df.dropna(subset=['home_goals', 'away_goals', 'result']).copy()
    done = done[(done.home_matches_before >= MIN_MATCHES) & (done.away_matches_before >= MIN_MATCHES)]
    done = done.sort_values(['date', 'match_id'])
    
    if len(done) < 300:
        raise ValueError('Se necesitan al menos 300 partidos completados con suficiente historial.')
        
    cut = int(len(done) * 0.80)
    train = done.iloc[:cut]
    cal = done.iloc[cut:]
    
    pois = fit_poisson_goals(train)
    gb, gbcols = build_classifier(train)
    lr, lcols = build_logistic(train)
    
    def get_raw_predictions(d):
        lh, la = predict_lambdas(pois, d)
        pp = make_poisson_probs_vectorized(lh, la)
        pg = align_classifier_probs(gb, d[gbcols])
        pl = align_classifier_probs(lr, d[lcols])
        pm = market_probs(d)
        
        pe = np.column_stack([
            d.elo_prob_home.values, 
            (1 - d.elo_prob_home.values) * 0.28, 
            (1 - d.elo_prob_home.values) * 0.72
        ])
        pe = normalize_probs(pe)
        return [pp, pg, pl, pe, pm], (lh, la)
        
    # Calibración
    plist_cal, _ = get_raw_predictions(cal)
    w, _ = optimize_weights(plist_cal, cal.result.values)
    
    pcal = sum(w[i] * plist_cal[i] for i in range(len(w)))
    temp = fit_temperature(pcal, cal.result.values)
    pcal = softmax(np.log(np.clip(pcal, 1e-8, 1.0)) / temp, axis=1)
    
    metrics = evaluate(pcal, cal.result.values)
    
    # Predicción final sobre todo el dataset
    allrows = df.copy()
    plist_all, lams = get_raw_predictions(allrows)
    p_all = sum(w[i] * plist_all[i] for i in range(len(w)))
    p_all = softmax(np.log(np.clip(p_all, 1e-8, 1.0)) / temp, axis=1)
    
    out = allrows.copy()
    out['lambda_home'], out['lambda_away'] = lams
    out[['p_model_home', 'p_model_draw', 'p_model_away']] = p_all
    out['model_entropy'] = -(p_all * np.log(np.clip(p_all, 1e-12, 1.0))).sum(axis=1)
    out['market_home'], out['market_draw'], out['market_away'] = market_probs(out).T
    
    meta = {
        'weights': w.tolist(),
        'temperature': temp,
        'calibration_metrics': metrics,
        'train_end': str(train.date.max()),
        'calibration_start': str(cal.date.min()),
        'n_train': len(train),
        'n_calibration': len(cal)
    }
    
    MODELS.mkdir(exist_ok=True)
    joblib.dump({
        'poisson': pois, 'gb': gb, 'gbcols': gbcols, 
        'lr': lr, 'lcols': lcols, 'weights': w, 'temperature': temp
    }, MODELS / 'ensemble.joblib')
    
    (MODELS / 'model_metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/processed/asof_features.csv')
    ap.add_argument('--output', default='data/processed/predictions_ensemble.csv')
    args = ap.parse_args()
    
    df = pd.read_csv(args.input, parse_dates=['date'])
    out, meta = train_and_predict(df)
    out.to_csv(args.output, index=False)
    
    print(json.dumps(meta, indent=2))
    print(out[['date', 'league', 'home_team', 'away_team', 'p_model_home', 'p_model_draw', 'p_model_away']].tail(20).to_string(index=False))


if __name__ == '__main__':
    main()