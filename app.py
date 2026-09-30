from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WinIA - Value Betting Dashboard", page_icon="⚽", layout="wide"
)


# Función para reemplazar "fixture" por el nombre de la liga de forma automática
def limpiar_fixture(df):
  league_col = next(
      (
          c
          for c in ["league", "division", "div", "competition", "competicion"]
          if c in df.columns
      ),
      None,
  )
  if league_col:
    for col in df.columns:
      mask = df[col].astype(str).str.lower() == "fixture"
      if mask.any():
        df.loc[mask, col] = df[league_col]
  return df


# ==========================================
# BARRA LATERAL (SIDEBAR) - GESTIÓN DE BANK
# ==========================================
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown(
    "💡 *Para actualizar datos y modelos, ejecuta el flujo en la pestaña Actions"
    " de GitHub.*"
)

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

# Cargar archivos de datos procesados
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
    df_val = limpiar_fixture(
        df_val
    )  # Limpia "fixture" usando la columna de liga

    if not df_val.empty:
      min_edge = st.slider(
          "Edge mínimo (%)", 0.0, 0.15, 0.025, 0.005, key="edge_slider"
      )
      df_filtered = df_val[df_val["edge"] >= min_edge].copy()

      if "stake" in df_filtered.columns:
        df_filtered["stake_ (€)"] = (
            df_filtered["stake"] * bankroll_inicial
        ).round(2)

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
        " actualización en GitHub Actions."
    )

with tab2:
  st.subheader("📅 Próximos Partidos y Predicciones del Modelo")
  if predictions_path.exists():
    df = pd.read_csv(predictions_path)
    df = limpiar_fixture(df)  # Limpia "fixture" usando la columna de liga

    if "date" in df.columns:
      df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")

      filtro_tiempo = st.selectbox(
          "Filtrar período:",
          [
              "Próximos y últimos 15 días",
              "Solo Próximos (Futuros)",
              "Solo últimos 7 días",
              "Ver últimos 100 partidos del registro",
          ],
          index=0,
          key="selectbox_tab2",
      )

      hoy = pd.Timestamp.now()

      if filtro_tiempo == "Próximos y últimos 15 días":
        hace_15_dias = hoy - timedelta(days=15)
        df_futuros = df[df["date_parsed"] >= hace_15_dias]
      elif filtro_tiempo == "Solo Próximos (Futuros)":
        df_futuros = df[df["date_parsed"] >= hoy]
      elif filtro_tiempo == "Solo últimos 7 días":
        hace_7_dias = hoy - timedelta(days=7)
        df_futuros = df[
            (df["date_parsed"] >= hace_7_dias) & (df["date_parsed"] <= hoy)
        ]
      else:
        df_futuros = df.tail(100)

      if "date_parsed" in df_futuros.columns:
        df_futuros = df_futuros.drop(columns=["date_parsed"])

      if not df_futuros.empty and "date" in df_futuros.columns:
        try:
          df_futuros = df_futuros.sort_values("date")
        except Exception:
          pass
    else:
      df_futuros = df.tail(100)

    st.write(f"Mostrando {len(df_futuros)} partidos:")
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
