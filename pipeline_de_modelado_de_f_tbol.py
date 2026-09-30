import argparse
import math
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# ==========================================
# 1. CÁLCULOS POISSON Y MULTIMERCADO
# ==========================================
def poisson_all_markets(lam_h, lam_a, max_goals=10):
    """Calcula la matriz de probabilidad de goles y deriva probabilidades

    para 1X2, Over/Under 2.5, BTTS (Ambos marcan) y AH0 (DNB).
    """
    probs = np.outer(
        poisson.pmf(np.arange(max_goals + 1), lam_h),
        poisson.pmf(np.arange(max_goals + 1), lam_a),
    )

    # Mercado 1X2
    ph = float(np.tril(probs, -1).sum())
    pd_prob = float(np.trace(probs))
    pa = float(np.triu(probs, 1).sum())
    z = ph + pd_prob + pa
    ph, pd_prob, pa = ph / z, pd_prob / z, pa / z

    # Mercado Over / Under 2.5
    p_under25 = float(
        sum(
            probs[i, j]
            for i in range(max_goals + 1)
            for j in range(max_goals + 1)
            if i + j <= 2
        )
    )
    p_over25 = 1.0 - p_under25

    # Mercado BTTS (Ambos marcan)
    p_btts_no = float(np.sum(probs[0, :]) + np.sum(probs[:, 0]) - probs[0, 0])
    p_btts_yes = 1.0 - p_btts_no

    # Mercado Handicap Asiatico 0 / DNB
    p_ah0_home = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    p_ah0_away = pa / (ph + pa) if (ph + pa) > 0 else 0.5

    return (
        ph,
        pd_prob,
        pa,
        p_over25,
        p_under25,
        p_btts_yes,
        p_btts_no,
        p_ah0_home,
        p_ah0_away,
    )


def compute_poisson_features(df, shrink=0.25):
    """Genera las probabilidades Poisson para todos los partidos del dataset."""
    league_mean = (
        df[["home_gf_5", "away_gf_5", "home_ga_5", "away_ga_5"]].stack().mean()
    )
    rows = []

    for _, r in df.iterrows():
        away_ga = r.get("away_ga_5")
        home_ga = r.get("home_ga_5")

        h_attack = (
            r.home_gf_5
            + (
                away_ga
                if (away_ga is not None and not np.isnan(away_ga))
                else league_mean
            )
        ) / 2
        a_attack = (
            r.away_gf_5
            + (
                home_ga
                if (home_ga is not None and not np.isnan(home_ga))
                else league_mean
            )
        ) / 2

        h_attack = (1 - shrink) * h_attack + shrink * league_mean
        a_attack = (1 - shrink) * a_attack + shrink * league_mean

        elo_diff = r.get("elo_diff_pre")
        if elo_diff is None or np.isnan(elo_diff):
            elo_diff = r.get("elo_home_pre", 1500) - r.get("elo_away_pre", 1500)

        elo_factor = np.clip(elo_diff / 400, -0.75, 0.75)

        lam_h = max(0.05, h_attack * math.exp(0.18 * elo_factor))
        lam_a = max(0.05, a_attack * math.exp(-0.18 * elo_factor))

        (
            p_h,
            p_d,
            p_a,
            p_over25,
            p_under25,
            p_btts_yes,
            p_btts_no,
            p_ah0_home,
            p_ah0_away,
        ) = poisson_all_markets(lam_h, lam_a)

        rows.append([
            lam_h,
            lam_a,
            p_h,
            p_d,
            p_a,
            p_over25,
            p_under25,
            p_btts_yes,
            p_btts_no,
            p_ah0_home,
            p_ah0_away,
        ])

    cols = [
        "lambda_home",
        "lambda_away",
        "p_poisson_home",
        "p_poisson_draw",
        "p_poisson_away",
        "p_over25",
        "p_under25",
        "p_btts_yes",
        "p_btts_no",
        "p_ah0_home",
        "p_ah0_away",
    ]
    res = pd.DataFrame(rows, columns=cols, index=df.index)
    return res


# ==========================================
# 2. ENSEMBLE Y CALIBRACIÓN DE MODELOS
# ==========================================
def prepare_features(df):
  """Prepara la matriz X de variables predictoras para ML de forma robusta

  para evitar valores constantes en partidos futuros.
  """
  feature_cols = [
      "elo_diff_pre",
      "elo_prob_home",
      "home_gf_5",
      "home_ga_5",
      "away_gf_5",
      "away_ga_5",
      "home_ppg_5",
      "away_ppg_5",
      "form_diff_5",
      "goal_diff_form_5",
      "rest_diff",
      "experience_diff",
      "lambda_home",
      "lambda_away",
      "p_poisson_home",
      "p_poisson_draw",
      "p_poisson_away",
  ]

  existing_cols = [c for c in feature_cols if c in df.columns]
  X = df[existing_cols].copy()

  # En lugar de una mediana global que clona las filas futuras,
  # rellenamos de forma inteligente con valores neutros lógicos de fútbol:
  # - Diferencias de Elo / forma a 0 (empate técnico de partida)
  # - Goles esperados a la media histórica estándar (ej. 1.25)
  fill_values = {
      "elo_diff_pre": 0.0,
      "elo_prob_home": 0.5,
      "home_gf_5": 1.25,
      "home_ga_5": 1.25,
      "away_gf_5": 1.25,
      "away_ga_5": 1.25,
      "home_ppg_5": 1.35,
      "away_ppg_5": 1.10,
      "form_diff_5": 0.0,
      "goal_diff_form_5": 0.0,
      "rest_diff": 0.0,
      "experience_diff": 0.0,
  }

  for col, val in fill_values.items():
    if col in X.columns:
      X[col] = X[col].fillna(val)

  # Cualquier otra columna numérica restante se puede rellenar con su mediana específica
  X = X.fillna(X.median())
  return X, existing_cols


def calibrate_probabilities(P, T=1.0):
    """Calibración por temperatura de matriz de probabilidades."""
    P_cal = np.power(P, 1.0 / T)
    P_cal /= P_cal.sum(axis=1, keepdims=True)
    return P_cal


# ==========================================
# 3. PIPELINE PRINCIPAL
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de modelado Ensemble WinIA"
    )
    parser.add_argument("--input", default="data/processed/asof_features.csv")
    parser.add_argument(
        "--output", default="data/processed/predictions_ensemble.csv"
    )
    parser.add_argument("--models-dir", default="data/models")
    args = parser.parse_args()

    print(f"Cargando dataset desde {args.input}...")
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Calcular variables Poisson y Multimercado
    print("Calculando modelos Poisson y métricas multimercado...")
    df_poisson = compute_poisson_features(df)
    for c in df_poisson.columns:
        df[c] = df_poisson[c]

    # 2. Separar datos de entrenamiento (partidos con resultado) y predicción
    df_train = df[
        df["result"].notna() & df["result"].isin(["H", "D", "A"])
    ].copy()
    df_train["target"] = df_train["result"].map(map_target).astype(int)

    X_train_full, feature_cols = prepare_features(df_train)
    y_train_full = df_train["target"].values

    # Split temporal para entrenamiento y validación/calibración (80% / 20%)
    split_idx = int(len(df_train) * 0.8)

    X_tr, y_tr = X_train_full.iloc[:split_idx], y_train_full[:split_idx]
    X_val, y_val = X_train_full.iloc[split_idx:], y_train_full[split_idx:]

    print(f"Entrenando modelos sobre {len(X_tr)} partidos pasados...")

    # A. Modelo Gradient Boosting
    gb = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=42
    )
    gb.fit(X_tr, y_tr)

    # B. Modelo Regresión Logística
    lr = LogisticRegression(max_iter=500, C=0.1, random_state=42)
    lr.fit(X_tr, y_tr)

    # C. Obtener predicciones en validación para ensamblar
    P_gb_val = gb.predict_proba(X_val)
    P_lr_val = lr.predict_proba(X_val)
    P_poi_val = df_train.iloc[split_idx:][
        ["p_poisson_home", "p_poisson_draw", "p_poisson_away"]
    ].values

    # Opciones Elo base
    if "elo_prob_home" in df_train.columns:
        p_elo_h = df_train.iloc[split_idx:]["elo_prob_home"].values
        p_elo_d = np.full_like(p_elo_h, 0.26)
        p_elo_a = 1.0 - p_elo_h - p_elo_d
        P_elo_val = np.column_stack([p_elo_h, p_elo_d, p_elo_a])
        P_elo_val = np.clip(P_elo_val, 0.01, 0.98)
        P_elo_val /= P_elo_val.sum(axis=1, keepdims=True)
    else:
        P_elo_val = P_poi_val

    # Optimizar pesos del Ensemble
    weights = optimize_weights(
        [P_gb_val, P_lr_val, P_poi_val, P_elo_val], y_val
    )
    print(
        f"Pesos optimizados del Ensemble -> GB: {weights[0]:.3f}, LR: {weights[1]:.3f}, Poisson: {weights[2]:.3f}, Elo: {weights[3]:.3f}"
    )

    # Re-entrenar modelos con el histórico completo
    print("Re-entrenando modelos con el histórico completo...")
    gb.fit(X_train_full, y_train_full)
    lr.fit(X_train_full, y_train_full)

    # Guardar modelos entrenados
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gb, models_dir / "gb_model.joblib")
    joblib.dump(lr, models_dir / "lr_model.joblib")

    # 3. GENERAR PREDICCIONES FINALES PARA TODO EL DATASET
    print("Generando predicciones de Ensemble para todos los partidos...")
    X_all, _ = prepare_features(df)

    P_gb_all = gb.predict_proba(X_all)
    P_lr_all = lr.predict_proba(X_all)
    P_poi_all = df[["p_poisson_home", "p_poisson_draw", "p_poisson_away"]].values

    if "elo_prob_home" in df.columns:
        p_elo_h_all = df["elo_prob_home"].values
        p_elo_d_all = np.full_like(p_elo_h_all, 0.26)
        p_elo_a_all = 1.0 - p_elo_h_all - p_elo_d_all
        P_elo_all = np.column_stack([p_elo_h_all, p_elo_d_all, p_elo_a_all])
        P_elo_all = np.clip(P_elo_all, 0.01, 0.98)
        P_elo_all /= P_elo_all.sum(axis=1, keepdims=True)
    else:
        P_elo_all = P_poi_all

    # Combinación lineal con los pesos optimizados
    P_ensemble = (
        weights[0] * P_gb_all
        + weights[1] * P_lr_all
        + weights[2] * P_poi_all
        + weights[3] * P_elo_all
    )

    # Asignar probabilidades 1X2 finales del modelo
    df["p_model_home"] = P_ensemble[:, 0]
    df["p_model_draw"] = P_ensemble[:, 1]
    df["p_model_away"] = P_ensemble[:, 2]

    # Normalizar y acotar probabilidades
    cols_prob = [
        "p_model_home",
        "p_model_draw",
        "p_model_away",
        "p_over25",
        "p_under25",
        "p_btts_yes",
        "p_btts_no",
        "p_ah0_home",
        "p_ah0_away",
    ]
    for c in cols_prob:
        df[c] = df[c].clip(0.001, 0.998)

    # Exportar archivo final procesado
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✓ Archivo generado exitosamente con {len(df)} filas: {out_path}")


if __name__ == "__main__":
    main()
