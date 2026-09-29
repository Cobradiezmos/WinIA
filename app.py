from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WinIA - Value Betting Dashboard", page_icon="⚽", layout="wide"
)

st.title("⚽ WinIA - Sistema Integral de Value Betting")
st.markdown(
    "Panel de control con oportunidades de valor multicriterio (1X2, Over/Under, BTTS y Hándicap)."
)

# Cargar candidatos de valor generados por el pipeline
candidates_path = Path("data/processed/value_candidates.csv")

if not candidates_path.exists():
  st.warning(
      "⚠️ No se encontró el archivo `value_candidates.csv`. Ejecuta la"
      " actualización diaria primero."
  )
else:
  df_val = pd.read_csv(candidates_path)

  if df_val.empty:
    st.info(
        "ℹ️ No hay oportunidades de valor que cumplan los criterios actuales"
        " en este momento."
    )
  else:
    st.subheader(
        f"🔥 Oportunidades de Valor Detectadas ({len(df_val)} apuestas)"
    )

    # Filtros laterales o superiores opcionales
    min_edge_filter = st.slider(
        "Filtrar por Edge mínimo (%)", 0.0, 0.15, 0.025, 0.005
    )
    df_filtered = df_val[df_val["edge"] >= min_edge_filter]

    # Mostrar tabla interactiva
    st.dataframe(
        df_filtered.sort_values(by="edge", ascending=False), use_container_width=True
    )

# Cargar también datos generales si se desea mostrar partidos futuros
predictions_path = Path("data/processed/predictions_ensemble.csv")
if not predictions_path.exists():
  predictions_path = Path("data/processed/predictions_elo_poisson.csv")

if predictions_path.exists():
  df_pred = pd.read_csv(predictions_path)
  st.subheader("📅 Próximos Partidos y Predicciones del Modelo")

  # Comprobación segura de columnas para evitar NameErrors o KeyErrors
  if "result" in df_pred.columns:
    df_futuros = df_pred[
        df_pred["result"].isna() | (df_pred["result"] == "")
    ]
  elif "home_goals" in df_pred.columns:
    df_futuros = df_pred[df_pred["home_goals"].isna()]
  else:
    df_futuros = df_pred.tail(20)  # Muestra los últimos si no hay filtro claro

  if not df_futuros.empty and "date" in df_futuros.columns:
    df_futuros = df_futuros.sort_values("date")

  st.dataframe(df_futuros, use_container_width=True)
