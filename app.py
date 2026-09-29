from pathlib import Path
import pandas as pd
import streamlit as st
import subprocess

st.set_page_config(
    page_title="WinIA - Value Betting Dashboard", page_icon="⚽", layout="wide"
)

# ==========================================
# BARRA LATERAL (SIDEBAR) - CONTROLES Y BANK
# ==========================================
st.sidebar.title("⚙️ Panel de Control")

# Botón para forzar actualización diaria de datos
if st.sidebar.button("🔄 Ejecutar Actualización Diaria"):
  with st.spinner("Actualizando datos y ejecutando modelos..."):
    try:
      result = subprocess.run(
          ["python", "daily_update.py"], capture_output=True, text=True, check=True
      )
      st.sidebar.success("¡Actualización completada con éxito!")
    except subprocess.CalledProcessError as e:
      st.sidebar.error(f"Error en la actualización: {e.stderr}")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Gestión de Bankroll")
bankroll_inicial = st.sidebar.number_input(
    "Bankroll Total (€)", min_value=10.0, value=1000.0, step=50.0
)
fraction_kelly = st.sidebar.slider(
    "Fracción de Kelly", min_value=0.1, max_value=1.0, value=0.25, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**WinIA v2.5** - Sistema Multicriterio (1X2, Over/Under, BTTS, AH0)"
)

# ==========================================
# VISTA PRINCIPAL
# ==========================================
st.title("⚽ WinIA - Sistema Integral de Value Betting")
st.markdown(
    "Panel de control para la gestión de apuestas de valor, seguimiento de"
    " modelos y optimización de capital."
)

# Cargar archivos de datos
candidates_path = Path("data/processed/value_candidates.csv")
predictions_path = Path("data/processed/predictions_ensemble.csv")
if not predictions_path.exists():
  predictions_path = Path("data/processed/predictions_elo_poisson.csv")

# Pestañas principales
tab1, tab2, tab3 = st.tabs(
    ["🔥 Oportunidades de Valor", "📅 Próximos Partidos", "📈 Histórico / Datos"]
)

with tab1:
  st.subheader("Oportunidades de Valor Detectadas")
  if candidates_path.exists():
    df_val = pd.read_csv(candidates_path)
    if not df_val.empty:
      # Filtros interactivos
      min_edge = st.slider(
          "Edge mínimo (%)", 0.0, 0.15, 0.025, 0.005, key="edge_slider"
      )
      df_filtered = df_val[df_val["edge"] >= min_edge].copy()

      # Calcular stake monetario real basado en el bankroll y kelly configurados en la sidebar
      if "stake" in df_filtered.columns:
        df_filtered["stake_ (€)"] = (
            df_filtered["stake"] * bankroll_inicial
        ).round(2)

      # Métricas rápidas de resumen
      col1, col2, col3 = st.columns(3)
      col1.metric("Apuestas Encontradas", len(df_filtered))
      col2.metric(
          "Edge Promedio",
          f"{(df_filtered['edge'].mean() * 100):.2f}%"
          if not df_filtered.empty
          else "0%",
      )
      col3.metric("Bankroll Configurado", f"{bankroll_inicial:,.2f} €")

      st.dataframe(
          df_filtered.sort_values(by="edge", ascending=False),
          use_container_width=True,
      )
    else:
      st.info("No hay oportunidades de valor con los filtros actuales.")
  else:
    st.warning(
        "⚠️ No se encontró el archivo de candidatos de valor. Ejecuta la"
        " actualización diaria."
    )

with tab2:
  st.subheader("Próximos Partidos y Predicciones del Modelo")
  if predictions_path.exists():
    df = pd.read_csv(predictions_path)

    # Arreglo seguro del error de la columna result
    if "result" in df.columns:
      df_futuros = df[df["result"].isna() | (df["result"] == "")]
    elif "home_goals" in df.columns:
      df_futuros = df[df["home_goals"].isna()]
    else:
      df_futuros = df

    if not df_futuros.empty and "date" in df_futuros.columns:
      df_futuros = df_futuros.sort_values("date")

    st.dataframe(df_futuros, use_container_width=True)
  else:
    st.warning("No se encontraron archivos de predicciones procesados.")

with tab3:
  st.subheader("Base de Datos y Registros de Sincronización")
  log_path = Path("data/logs/daily_sync_log.csv")
  if log_path.exists():
    st.write("Registro de sincronización diaria:")
    st.dataframe(pd.read_csv(log_path), use_container_width=True)
  else:
    st.info("No hay registros de sincronización disponibles.")
