from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.detectors import (
    detect_zscore_rolling,
    detect_iqr,
    detect_moving_average,
)

DATA_PATH = Path("data/processed/consumo_diario.csv")

st.set_page_config(
    page_title="Detecção de Anomalias",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


st.title("Detecção de Anomalias no Consumo de Energia")
st.caption(
    "Protótipo acadêmico — métodos estatísticos interpretáveis "
    "(Z-score em janela móvel, IQR e Média Móvel)"
)

if not DATA_PATH.exists():
    st.error(f"Arquivo não encontrado: {DATA_PATH}")
    st.stop()

df = load_data()

# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.header("Parâmetros")

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    periodo = st.date_input(
        "Período",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    metodo = st.selectbox(
        "Método",
        ["Z-score em janela móvel", "IQR", "Média móvel"],
    )

    window = None
    limiar = None
    k = None
    rel = None

    if metodo == "Z-score em janela móvel":
        window = st.slider("Janela (dias)", 3, 60, 30)
        limiar = st.slider("Limiar (|z|)", 1.0, 5.0, 3.0, 0.1)
    elif metodo == "IQR":
        k = st.slider("Fator k (IQR)", 0.5, 3.0, 1.5, 0.1)
    else:
        window = st.slider("Janela (dias)", 3, 60, 30)
        rel = st.slider("Limiar de desvio relativo", 0.10, 1.00, 0.30, 0.05)

    analisar = st.button("Analisar", type="primary")

# ------------------------- PERÍODO -------------------------
if isinstance(periodo, tuple) and len(periodo) == 2:
    d0 = pd.to_datetime(periodo[0])
    d1 = pd.to_datetime(periodo[1])
    if d0 > d1:
        st.warning("Período inválido: data inicial maior que a final.")
        st.stop()
    view = df[(df["date"] >= d0) & (df["date"] <= d1)].copy()
else:
    view = df.copy()

if view.empty:
    st.warning("Nenhum registro no período selecionado.")
    st.stop()

st.subheader("Dados")
c1, c2, c3 = st.columns(3)
c1.metric("Registros no período", len(view))
c2.metric("Dias completos", int((view["data_quality"] == "complete").sum()))
c3.metric("Dias parciais", int((view["data_quality"] == "partial").sum()))

# ------------------------- ANÁLISE -------------------------
if analisar:
    if metodo == "Z-score em janela móvel":
        res = detect_zscore_rolling(view, window=window, threshold=limiar)
        anom_col = "anomaly_zscore"
        score_col = "zscore"
    elif metodo == "IQR":
        res = detect_iqr(view, k=k)
        anom_col = "anomaly_iqr"
        score_col = None
    else:
        res = detect_moving_average(view, window=window, rel_threshold=rel)
        anom_col = "anomaly_ma"
        score_col = "ma_rel_dev"

    res[anom_col] = res[anom_col].astype(bool)

    total = len(res)
    n_anom = int(res[anom_col].sum())
    taxa = (n_anom / total * 100) if total else 0.0

    st.subheader("Resultados")
    m1, m2, m3 = st.columns(3)
    m1.metric("Registros analisados", total)
    m2.metric("Anomalias detectadas", n_anom)
    m3.metric("Taxa de anomalias", f"{taxa:.2f}%")

    st.caption(
        "Os pontos sinalizados são **candidatos a anomalia** "
        "(não há rótulo real neste modo)."
    )

    # ---------- gráfico ----------
    anom = res[res[anom_col]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=res["date"],
            y=res["consumption_kwh"],
            mode="lines",
            name="Consumo diário (kWh)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=anom["date"],
            y=anom["consumption_kwh"],
            mode="markers",
            name="Anomalia",
            marker=dict(color="red", size=9, symbol="x"),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Data",
        yaxis_title="Consumo (kWh)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- alertas ----------
    st.subheader("Alertas")

    if anom.empty:
        st.info("Nenhuma anomalia detectada com os parâmetros atuais.")
    else:
        alert = anom[["date", "consumption_kwh"]].copy()

        if score_col is not None and score_col in anom.columns:
            alert[score_col] = anom[score_col].values

        if metodo == "Z-score em janela móvel":
            alert["tipo"] = [
                "Pico" if z >= 0 else "Queda" for z in anom["zscore"].fillna(0)
            ]
        elif metodo == "IQR":
            alert["tipo"] = [
                "Pico" if v > u else "Queda"
                for v, u in zip(anom["consumption_kwh"], anom["iqr_upper"])
            ]
        else:
            alert["tipo"] = [
                "Pico" if v > m else "Queda"
                for v, m in zip(anom["consumption_kwh"], anom["ma_mean"])
            ]

        alert = alert.sort_values("date")
        st.dataframe(alert, use_container_width=True)