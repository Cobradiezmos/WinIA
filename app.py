from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WinIA - Value Betting Dashboard", page_icon="⚽", layout="wide"
)

st.title("⚽ WinIA - Sistema Integral de Value Betting")
st.markdown(
    "Panel de control para la gestión de apuestas de valor, seguimiento de"
    " modelos y backtest."
)

# Cargar candidatos de valor
candidates_path = Path("data/processed/value_candidates.csv")
predictions_path = Path("data/processed/predictions_ensemble.csv")
if not predictions_path.exists():
  predictions_path = Path("data/processed/predictions_elo_poisson.csv")

# Pestañas o secciones principales (recuperando tu vista original)
tab1, tab2, tab3 = st.tabs(
    ["🔥 Oportunidades de Valor", "📅 Próximos Partidos", "📈 Histórico / Datos"]
)

with tab1:
  st.subheader("Oportunidades de Valor Detectadas")
  if candidates_path.exists():
    df_val = pd.read_csv(candidates_path)
    if not df_val.empty:
      # Filtros interactivos originales
      min_edge = st.slider(
          "Edge mínimo (%)", 0.0, 0.15, 0.025, 0.005, key="edge_slider"
      )
      df_filtered = df_val[df_val["edge"] >= min_edge]
      st.dataframe(
          df_filtered.sort_values(by="edge", ascending=False),
          use_container_width=True,
      )
    else:
      st.info("No hay oportunidades de valor con los filtros actuales.")
  else:
  data_file = Path("data/processed/predictions_ensemble.csv")
  if not data_file.exists():
    data_file = Path("data/processed/predictions_elo_poisson.csv")

  if data_file.exists():
    df = pd.read_csv(data_file)
    
    # Arreglo seguro del error de la columna result asegurando que no rompa la app
    if "result" in df.columns:
      df_futuros = df[df["result"].isna() | (df["result"] == "")]
    elif "home_goals" in df.columns:
      df_futuros = df[df["home_goals"].isna()]
    else:
      df_futuros = df
      
    st.dataframe(df_futuros, use_container_width=True)
  else:
    st.warning("No se encontraron archivos de predicciones procesados.")

with tab3:
  st.subheader("Base de Datos y Registros")
  log_path = Path("data/logs/daily_sync_log.csv")
  if log_path.exists():
    st.write("Registro de sincronización diaria:")
    st.dataframe(pd.read_csv(log_path), use_container_width=True)
  else:
    st.info("No hay registros de sincronización disponibles.")
