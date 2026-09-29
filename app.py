import json
import subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Value Betting Model", layout="wide")

st.title("⚽ Football Value Betting Dashboard")

# Barra lateral
st.sidebar.header("⚙️ Configuración")
bankroll = st.sidebar.number_input("Banca Total ($)", value=1000.0, step=100.0)
kelly_frac = st.sidebar.slider("Fracción de Kelly", 0.05, 1.0, 0.25, step=0.05)
min_edge = st.sidebar.slider("Edge Mínimo", 0.0, 0.10, 0.025, step=0.005)
min_ev = st.sidebar.slider("EV Mínimo", 0.0, 0.10, 0.02, step=0.005)

if st.sidebar.button("🔄 Actualizar y Re-entrenar"):
    with st.spinner("Ejecutando pipeline diario..."):
        subprocess.run(["python", "daily_update.py"], check=True)
    st.success("¡Modelo actualizado!")

# Pestañas principales
tab1, tab2 = st.tabs(["🎯 Oportunidades de Valor", "📈 Backtest Historico"])

with tab1:
    st.header("Candidatos de Valor")
    val_file = Path("data/processed/value_candidates.csv")
    if val_file.exists():
        df_val = pd.read_csv(val_file)
        filtered = df_val[(df_val['edge'] >= min_edge) & (df_val['ev'] >= min_ev)].copy()
        if not filtered.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Partidos Encontrados", len(filtered))
            col2.metric("EV Medio", f"{filtered['ev'].mean()*100:.2f}%")
            col3.metric("Riesgo Total", f"${filtered['stake'].sum():.2f}")
            st.dataframe(filtered, use_container_width=True)
        else:
            st.info("No hay oportunidades con los filtros actuales.")
    else:
        st.warning("No se encontraron predicciones. Haz clic en 'Actualizar y Re-entrenar'.")

with tab2:
    st.header("Resultados de Backtest")
    met_file = Path("data/processed/backtest/metrics.json")
    bets_file = Path("data/processed/backtest/bets.csv")
    if met_file.exists() and bets_file.exists():
        with open(met_file) as f:
            m = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI", f"{m.get('roi', 0)*100:.2f}%")
        c2.metric("Beneficio", f"${m.get('profit', 0):.2f}")
        c3.metric("Aciertos", f"{m.get('win_rate', 0)*100:.1f}%")
        c4.metric("Max Drawdown", f"${m.get('max_drawdown', 0):.2f}")
        
        df_b = pd.read_csv(bets_file)
        if 'cum_profit' in df_b.columns:
            st.line_chart(df_b.set_index('date')['cum_profit'])