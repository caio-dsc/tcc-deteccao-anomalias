import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# garante que "src" seja encontrado quando o Streamlit executa esta página
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors import (
    detect_zscore_rolling,
    detect_iqr,
    detect_moving_average,
)
from src.injection import inject_anomalies
from src.metrics import (
    confusion_counts,
    precision,
    recall,
    f1_score,
    false_positive_rate,
)

DATA_PATH = Path("data/processed/consumo_diario.csv")

st.set_page_config(
    page_title="Avaliação dos Métodos",
    page_icon="📊",
    layout="wide",
)

st.title("Avaliação dos Métodos (anomalias injetadas)")
st.caption(
    "Nesta página as anomalias são **injetadas de forma controlada**, "
    "gerando rótulos verdadeiros para cálculo de métricas."
)

if not DATA_PATH.exists():
    st.error(f"Arquivo não encontrado: {DATA_PATH}")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _evaluate(y_true, y_pred):
    c = confusion_counts(y_true, y_pred)
    p = precision(c["tp"], c["fp"])
    r = recall(c["tp"], c["fn"])
    return {
        "tp": c["tp"],
        "fp": c["fp"],
        "tn": c["tn"],
        "fn": c["fn"],
        "precision": p,
        "recall": r,
        "f1": f1_score(p, r),
        "fpr": false_positive_rate(c["fp"], c["tn"]),
    }


# ------------------------- SIDEBAR -------------------------
with st.sidebar:
    st.header("Experimento")

    n_anom = st.slider("Quantidade de anomalias injetadas", 5, 100, 40)
    spike_ratio = st.slider("Proporção de picos (spike)", 0.0, 1.0, 0.5, 0.1)
    seed = st.number_input("Seed (reprodutibilidade)", value=42, step=1)
    intensity = st.selectbox("Intensidade", ["moderate", "strong"])

    st.subheader("Parâmetros dos métodos")
    z_window = st.slider("Z-score: janela", 3, 60, 30)
    z_thr = st.slider("Z-score: limiar", 1.0, 5.0, 3.0, 0.1)
    k_iqr = st.slider("IQR: fator k", 0.5, 3.0, 1.5, 0.1)
    ma_window = st.slider("Média móvel: janela", 3, 60, 30)
    ma_thr = st.slider("Média móvel: limiar relativo", 0.10, 1.00, 0.30, 0.05)

    executar = st.button("Executar avaliação", type="primary")


# ------------------------- EXECUÇÃO -------------------------
if executar:
    df = load_data()

    try:
        df_inj = inject_anomalies(
            df,
            n=int(n_anom),
            spike_ratio=float(spike_ratio),
            seed=int(seed),
            min_index=30,
            intensity=intensity,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    rows = []

    # Z-score
    out = detect_zscore_rolling(df_inj, window=int(z_window), threshold=float(z_thr))
    mask = (
        (out["data_quality"] == "complete")
        & out["rolling_mean"].notna()
        & out["rolling_std"].notna()
        & (out["rolling_std"] > 0)
    )
    m = _evaluate(out.loc[mask, "label"], out.loc[mask, "anomaly_zscore"])
    rows.append(
        {
            "method": "Z-score",
            "params": f"window={z_window}, thr={z_thr}",
            "n_avaliado": int(mask.sum()),
            **m,
        }
    )

    # IQR
    out = detect_iqr(df_inj, k=float(k_iqr))
    mask = out["data_quality"] == "complete"
    m = _evaluate(out.loc[mask, "label"], out.loc[mask, "anomaly_iqr"])
    rows.append(
        {
            "method": "IQR",
            "params": f"k={k_iqr}",
            "n_avaliado": int(mask.sum()),
            **m,
        }
    )

    # Média móvel
    out = detect_moving_average(
        df_inj, window=int(ma_window), rel_threshold=float(ma_thr)
    )
    mask = (
        (out["data_quality"] == "complete")
        & out["ma_mean"].notna()
        & (out["ma_mean"] > 0)
    )
    m = _evaluate(out.loc[mask, "label"], out.loc[mask, "anomaly_ma"])
    rows.append(
        {
            "method": "Média móvel",
            "params": f"window={ma_window}, rel={ma_thr}",
            "n_avaliado": int(mask.sum()),
            **m,
        }
    )

    tbl = pd.DataFrame(rows)
    for col in ["precision", "recall", "f1", "fpr"]:
        tbl[col] = tbl[col].round(3)

    st.subheader("Métricas")
    st.dataframe(tbl, use_container_width=True)

    st.subheader("Comparativo (F1-score)")
    fig = go.Figure(
        go.Bar(x=tbl["method"], y=tbl["f1"], marker_color="steelblue")
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Matrizes de confusão"):
        for _, row in tbl.iterrows():
            st.write(f"**{row['method']}** — {row['params']}")
            cm = pd.DataFrame(
                {
                    "Previsto +": [int(row["tp"]), int(row["fp"])],
                    "Previsto -": [int(row["fn"]), int(row["tn"])],
                },
                index=["Real + (anomalia)", "Real - (normal)"],
            )
            st.dataframe(cm, use_container_width=True)