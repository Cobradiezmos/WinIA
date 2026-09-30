from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="WinIA - Value Betting Dashboard", page_icon="⚽", layout="wide"
)


# Función específica para limpiar "fixtures.csv" en partidos nuevos
def limpiar_fixtures_nuevos(df):
  league_cols = [
      c
      for c in [
          "league",
          "division",
          "div",
          "competition",
          "competicion",
          "League",
          "Division",
      ]
      if c in df.columns
  ]

  for col in df.columns:
    mask = df[col].astype(str).str.lower().str.contains("fixtures.csv", na=False)
    if mask.any():
      reemplazado = False
      for l_col in league_cols:
        if l_col != col:
          df.loc[mask, col] = df[l_col]
          reemplazado = True
          break
      if not reemplazado:
        df.loc[mask, col] = "Próxima Jornada"
  return df


# ==========================================
# BARRA LATERAL (SIDEBAR) - GESTIÓN Y ACCIONES
# ==========================================
st.sidebar.title("⚙️ Panel de Control")
st.sidebar.markdown(
    "💡 *Los flujos pesados se ejecutan en segundo plano mediante GitHub"
    " Actions.*"
)

# Botón de actualización remota vía GitHub Actions
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Sincronización Remota")
github_token = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
# REEMPLAZA "tu-usuario/tu-repositorio" por tu usuario y repo real de GitHub
repo_name = "Cobradiezmos/WinIA"

if st.sidebar.button("🚀 Actualizar Datos (GitHub)"):
  if not github_token:
    st.sidebar.error("⚠️ Configura el GITHUB_TOKEN en los secretos de Streamlit.")
  else:
    url = f"https://api.github.com/repos/{repo_name}/actions/workflows/daily_update.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    data = {"ref": "main"}
    try:
      response = requests.post(url, headers=headers, json=data)
      if response.status_code == 204:
        st.sidebar.success(
            "✅ ¡Orden enviada! GitHub Actions está actualizando los datos."
        )
      else:
        st.sidebar.error(
            f"❌ Error al conectar con GitHub: {response.status_code}"
        )
    except Exception as e:
      st.sidebar.error(f"❌ Error de red: {e}")

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
  st.subheader("Oportunidades de Valor Detectadas (Estrategia Low-Risk)")
  if candidates_path.exists():
    df_val = pd.read_csv(candidates_path)
    df_val = limpiar_fixtures_nuevos(df_val)

    if not df_val.empty:
      # Filtros para acotar riesgo (Edge y Cuota Máxima)
      col_f1, col_f2 = st.columns(2)
      with col_f1:
        min_edge = st.slider(
            "Edge mínimo (%)", 0.0, 0.15, 0.025, 0.005, key="edge_slider"
        )

      with col_f2:
        odd_col = next(
            (c for c in ["odd", "odds", "cuota", "price"] if c in df_val.columns),
            None,
        )
        if odd_col:
          max_cuota = st.slider(
              "Cuota máxima (Control de riesgo)",
              float(df_val[odd_col].min())
              if not df_val[odd_col].empty
              else 1.0,
              float(min(20.0, df_val[odd_col].max()))
              if not df_val[odd_col].empty
              else 10.0,
              2.5,
              0.1,
              key="max_odd_slider",
          )
        else:
          max_cuota = None

      # Aplicación de filtros
      df_filtered = df_val[df_val["edge"] >= min_edge].copy()
      if odd_col and max_cuota:
        df_filtered = df_filtered[df_filtered[odd_col] <= max_cuota]

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
    df = limpiar_fixtures_nuevos(df)

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
