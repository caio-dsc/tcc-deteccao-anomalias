import numpy as np
import pandas as pd


def inject_spike(
    df: pd.DataFrame,
    idx: int,
    factor: float = 2.0,
    label_col: str = "label",
    type_col: str = "injected_type",
) -> pd.DataFrame:
    """
    Injeta um pico multiplicando o consumo por um fator.
    Marca label=1 e injected_type="spike".
    """
    out = df.copy()
    out["consumption_kwh"] = out["consumption_kwh"].astype(float)

    if label_col not in out.columns:
        out[label_col] = 0
    if type_col not in out.columns:
        out[type_col] = pd.NA

    out.loc[idx, "consumption_kwh"] = out.loc[idx, "consumption_kwh"] * factor
    out.loc[idx, label_col] = 1
    out.loc[idx, type_col] = "spike"

    return out


def inject_drop(
    df: pd.DataFrame,
    idx: int,
    factor: float = 0.25,
    label_col: str = "label",
    type_col: str = "injected_type",
) -> pd.DataFrame:
    """
    Injeta uma queda multiplicando o consumo por um fator (< 1).
    Marca label=1 e injected_type="drop".
    """
    out = df.copy()
    out["consumption_kwh"] = out["consumption_kwh"].astype(float)

    if label_col not in out.columns:
        out[label_col] = 0
    if type_col not in out.columns:
        out[type_col] = pd.NA

    out.loc[idx, "consumption_kwh"] = out.loc[idx, "consumption_kwh"] * factor
    out.loc[idx, label_col] = 1
    out.loc[idx, type_col] = "drop"

    return out


def inject_anomalies(
    df: pd.DataFrame,
    n: int = 20,
    spike_ratio: float = 0.5,
    seed: int = 42,
    min_index: int = 30,
    intensity: str = "moderate",
    label_col: str = "label",
    type_col: str = "injected_type",
) -> pd.DataFrame:
    """
    Injeta anomalias (spike/drop) em dias completos e cria rótulos.

    Regras:
    - Só injeta em dias com data_quality == "complete".
    - Só injeta a partir de min_index (para não atrapalhar detectores com janela).
    - Reprodutível via seed.
    - intensity: "moderate" ou "strong".
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    # garante float (drops podem gerar valores fracionários)
    out["consumption_kwh"] = out["consumption_kwh"].astype(float)

    if label_col not in out.columns:
        out[label_col] = 0
    else:
        out[label_col] = 0  # zera por segurança

    if type_col not in out.columns:
        out[type_col] = pd.NA
    else:
        out[type_col] = pd.NA

    if intensity not in ("moderate", "strong"):
        raise ValueError("intensity must be 'moderate' or 'strong'")

    if intensity == "moderate":
        spike_factor = 1.3
        drop_factor = 0.7
    else:  # strong
        spike_factor = 2.0
        drop_factor = 0.25

    # candidatos: completos e após min_index
    valid = (out["data_quality"] == "complete") & out["consumption_kwh"].notna()
    candidate_idx = out.index[valid].to_numpy()
    candidate_idx = candidate_idx[candidate_idx >= min_index]

    if n <= 0:
        return out

    if len(candidate_idx) < n:
        raise ValueError(
            f"Not enough candidate days to inject. "
            f"Candidates={len(candidate_idx)} requested n={n}. "
            f"Try reducing n or min_index."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidate_idx, size=n, replace=False)

    n_spikes = int(round(n * spike_ratio))
    n_spikes = max(0, min(n, n_spikes))
    n_drops = n - n_spikes

    perm = rng.permutation(chosen)
    spike_idx = perm[:n_spikes]
    drop_idx = perm[n_spikes:]

    # aplica spikes
    out.loc[spike_idx, "consumption_kwh"] = out.loc[spike_idx, "consumption_kwh"] * spike_factor
    out.loc[spike_idx, label_col] = 1
    out.loc[spike_idx, type_col] = "spike"

    # aplica drops
    out.loc[drop_idx, "consumption_kwh"] = out.loc[drop_idx, "consumption_kwh"] * drop_factor
    out.loc[drop_idx, label_col] = 1
    out.loc[drop_idx, type_col] = "drop"

    return out