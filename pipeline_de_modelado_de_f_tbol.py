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
  """Calcula la matriz de probabilidad de goles y deriva probabilidades para 1X2, Over/Under 2.5, BTTS (Ambos marcan) y AH0 (DNB)."""
  if np.isnan(lam_h) or lam_h <= 0:
    lam_h = 1.25
  if np.isnan(lam_a) or lam_a <= 0:
    lam_a = 1.10

  probs = np.outer(
      poisson.pmf(np.arange(max_goals + 1), lam_h),
      poisson.pmf(np.arange(max_goals + 1), lam_a),
  )

  # Mercado 1X2
  ph = float(np.tril(probs, -1).sum())
  pd_prob = float(np.trace(probs))
  pa = float(np.triu(probs, 1).sum())
  z = ph + pd_prob + pa

  if z > 0:
    ph, pd_prob, pa = ph / z, pd_prob / z, pa / z
  else:
    ph, pd_prob, pa = 0.45, 0.28, 0.27

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
  """Genera las probabilidades Poisson para todos los partidos del dataset de forma blindada ante nulos."""
  cols_mean = ["home_gf_5", "away_gf_5", "home_ga_5", "away_ga_5"]
  existing = [c for c in cols_mean if c in df.columns]

  if existing and not df[existing].dropna(how="all").empty:
    league_mean = df[existing].stack().mean()
    if np.isnan(league_mean):
      league_mean = 1.25
  else:
    league_mean = 1.25

  rows = []

  for _, r in df.iterrows():
    home_gf = r.get("home_gf_5")
    away_gf = r.get("away_gf_5")
    home_ga = r.get("home_ga_5")
    away_ga = r.get("away_ga_5")

    h_gf = home_gf if pd.notna(home_gf) else league_mean
    a_gf = away_gf if pd.notna(away_gf) else league_mean
    h_ga = home_ga if pd.notna(home_ga) else league_mean
    a_ga = away_ga if pd.notna(away_ga) else league_mean

    h_attack = (h_gf + a_ga) / 2.0
    a_attack = (a_gf + h_ga) / 2.0

    h_attack = (1.0 - shrink) * h_attack + shrink * league_mean
    a_attack = (1.0 - shrink) * a_attack + shrink * league_mean

    elo_diff = r.get("elo_diff_pre")
    if pd.isna(elo_diff):
      elo_h = r.get("elo_home_pre", 1500)
      elo_a = r.get("elo_away_pre", 1500)
      elo_h = elo_h if pd.notna(elo_h) else 1500.0
      elo_a = elo_a if pd.notna(elo_a) else 1500.0
      elo_diff = elo_h - elo_a

    elo_factor = np.clip(elo_diff / 400.0, -0.75, 0.75)

    lam_h = max(0.05, h_attack * math.exp(0.18 * elo_factor))
    lam_a = max(0.05, a_attack * math.exp(-0.18 * elo_factor))

    if np.isnan(lam_h):
      lam_h = 1.25
    if np.isnan(lam_a):
      lam_a = 1.10

    res = poisson_all_markets(lam_h, lam_a)
    rows.append([lam_h, lam_a] + list(res))

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
  return pd.DataFrame(rows, columns=cols, index=df.index)


# ==========================================
# 2. ENSEMBLE Y CALIBRACIÓN DE MODELOS
# ==========================================
def prepare_features(df):
  """Prepara la matriz X de variables predictoras para ML de forma completamente segura."""
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
  X = X.fillna(X.median()).fillna(0.0)
  return X, existing_cols


def map_target(result):
  if result == "H":
    return 0
  if result == "D":
    return 1
  if result == "A":
    return 2
  return np.nan


def optimize_weights(P_list, y_true):
  """Optimización de ponderación del Ensemble minimizando Log Loss."""
  P_list = [np.nan_to_num(P, nan=0.3333) for P in P_list]
  n_models = len(P_list)

  def loss(weights):
    weights = weights / np.sum(weights)
    P_ens = sum(w * P for w, P in zip(weights, P_list))
    P_ens = np.clip(P_ens, 1e-5, 1 - 1e-5)
    one_hot = np.eye(3)[y_true]
    return -np.mean(np.sum(one_hot * np.log(P_ens), axis=1))

  init_weights = np.ones(n_models) / n_models
  bounds = [(0, 1)] * n_models
  res = minimize(loss, init_weights, bounds=bounds, method="SLSQP")
  if res.success and not np.isnan(res.x).any():
    final_w = res.x / np.sum(res.x)
  else:
    final_w = init_weights
  return final_w


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
    p_elo_h = np.nan_to_num(p_elo_h, nan=0.45)
    p_elo_d = np.full_like(p_elo_h, 0.26)
    p_elo_a = 1.0 - p_elo_h - p_elo_d
    P_elo_val = np.column_stack([p_elo_h, p_elo_d, p_elo_a])
    P_elo_val = np.clip(P_elo_val, 0.01, 0.98)
    P_elo_val /= P_elo_val.sum(axis=1, keepdims=True)
  else:
    P_elo_val = P_poi_val

  # Optimizar pesos del Ensemble
  weights = optimize_weights([P_gb_val, P_lr_val, P_poi_val, P_elo_val], y_val)
  print(
      f"Pesos optimizados del Ensemble -> GB: {weights[0]:.3f}, LR:"
      f" {weights[1]:.3f}, Poisson: {weights[2]:.3f}, Elo: {weights[3]:.3f}"
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
  P_poi_all = np.nan_to_num(P_poi_all, nan=0.3333)

  if "elo_prob_home" in df.columns:
    p_elo_h_all = df["elo_prob_home"].values
    p_elo_h_all = np.nan_to_num(p_elo_h_all, nan=0.45)
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
    if c in df.columns:
      df[c] = df[c].fillna(0.3333).clip(0.001, 0.998)

  # Exportar archivo final procesado
  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_path, index=False)

  print(f"✓ Archivo generado exitosamente con {len(df)} filas: {out_path}")


if __name__ == "__main__":
  main()
