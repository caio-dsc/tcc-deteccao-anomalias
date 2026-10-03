from pathlib import Path

import pandas as pd

from src.detectors import detect_zscore_rolling, detect_iqr, detect_moving_average
from src.injection import inject_anomalies
from src.metrics import confusion_counts, precision, recall, f1_score, false_positive_rate


DATA_PATH = Path("data/processed/consumo_diario.csv")
OUT_DIR = Path("results")
OUT_CSV = OUT_DIR / "resultados.csv"


def _evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    p = precision(tp, fp)
    r = recall(tp, fn)
    f1 = f1_score(p, r)
    fpr = false_positive_rate(fp, tn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "fpr": fpr,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. Gere com process_data.py ou adicione o CSV processado."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) carrega base
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2) injeta anomalias (gera label e injected_type)
    # Ajuste esses parâmetros se quiser:
    df_inj = inject_anomalies(
        df,
        n=40,
        spike_ratio=0.5,
        seed=42,
        min_index=30,
        intensity="moderate",
    )

    results_rows = []

    # =========================
    # Z-SCORE (rolling)
    # =========================
    z_cfgs = [
        {"window": 7, "threshold": 3.0},
        {"window": 30, "threshold": 3.0},
    ]

    for cfg in z_cfgs:
        out = detect_zscore_rolling(df_inj, window=cfg["window"], threshold=cfg["threshold"])

        # avalia somente onde o método é aplicável
        mask = (
            (out["data_quality"] == "complete")
            & out["rolling_mean"].notna()
            & out["rolling_std"].notna()
            & (out["rolling_std"] > 0)
        )

        y_true = out.loc[mask, "label"]
        y_pred = out.loc[mask, "anomaly_zscore"]

        m = _evaluate(y_true, y_pred)
        results_rows.append(
            {
                "method": "zscore_rolling",
                "params": f"window={cfg['window']}, threshold={cfg['threshold']}",
                "n_total": int(mask.sum()),
                "n_injected": int(y_true.sum()),
                **m,
            }
        )

    # =========================
    # IQR
    # =========================
    iqr_cfgs = [{"k": 1.5}]

    for cfg in iqr_cfgs:
        out = detect_iqr(df_inj, k=cfg["k"])

        mask = (out["data_quality"] == "complete")

        y_true = out.loc[mask, "label"]
        y_pred = out.loc[mask, "anomaly_iqr"]

        m = _evaluate(y_true, y_pred)
        results_rows.append(
            {
                "method": "iqr",
                "params": f"k={cfg['k']}",
                "n_total": int(mask.sum()),
                "n_injected": int(y_true.sum()),
                **m,
            }
        )

    # =========================
    # MOVING AVERAGE
    # =========================
    ma_cfgs = [
        {"window": 7, "rel_threshold": 0.3},
        {"window": 30, "rel_threshold": 0.3},
    ]

    for cfg in ma_cfgs:
        out = detect_moving_average(
            df_inj,
            window=cfg["window"],
            rel_threshold=cfg["rel_threshold"],
        )

        mask = (
            (out["data_quality"] == "complete")
            & out["ma_mean"].notna()
            & (out["ma_mean"] > 0)
        )

        y_true = out.loc[mask, "label"]
        y_pred = out.loc[mask, "anomaly_ma"]

        m = _evaluate(y_true, y_pred)
        results_rows.append(
            {
                "method": "moving_average",
                "params": f"window={cfg['window']}, rel_threshold={cfg['rel_threshold']}",
                "n_total": int(mask.sum()),
                "n_injected": int(y_true.sum()),
                **m,
            }
        )

    # 3) salva resultados
    results_df = pd.DataFrame(results_rows)

    # ordena por f1 desc (só para facilitar leitura)
    results_df = results_df.sort_values(["f1", "precision", "recall"], ascending=False).reset_index(drop=True)

    results_df.to_csv(OUT_CSV, index=False)

    print("\n=== AVALIAÇÃO (INJEÇÃO + MÉTRICAS) ===")
    print(f"Base usada: {DATA_PATH}")
    print(f"Resultados salvos em: {OUT_CSV}\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()