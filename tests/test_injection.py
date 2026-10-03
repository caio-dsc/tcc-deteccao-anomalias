import pandas as pd

from src.injection import inject_spike, inject_drop, inject_anomalies


def _make_df(consumptions, qualities=None, start="2024-01-01"):
    if qualities is None:
        qualities = ["complete"] * len(consumptions)

    dates = pd.date_range(start=start, periods=len(consumptions), freq="D")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "consumption_kwh": consumptions,
            "data_quality": qualities,
        }
    )


def test_inject_spike_changes_value_and_sets_label():
    df = _make_df([10, 10, 10, 10])
    out = inject_spike(df, idx=2, factor=2.0)

    assert out.loc[2, "consumption_kwh"] == 20
    assert out.loc[2, "label"] == 1
    assert out.loc[2, "injected_type"] == "spike"

    # demais não mudam
    assert out.loc[0, "label"] == 0
    assert out.loc[0, "consumption_kwh"] == 10


def test_inject_drop_changes_value_and_sets_label():
    df = _make_df([10, 10, 10, 10])
    out = inject_drop(df, idx=1, factor=0.25)

    assert out.loc[1, "consumption_kwh"] == 2.5
    assert out.loc[1, "label"] == 1
    assert out.loc[1, "injected_type"] == "drop"

    # demais não mudam
    assert out.loc[0, "label"] == 0
    assert out.loc[0, "consumption_kwh"] == 10


def test_inject_anomalies_inserts_exactly_n_labels_and_respects_rules():
    # 40 dias: primeiros 10 completos, depois alguns partial misturados
    consumptions = [10] * 40
    qualities = ["complete"] * 40
    qualities[5] = "partial"
    qualities[12] = "partial"
    qualities[25] = "partial"

    df = _make_df(consumptions, qualities=qualities)

    out = inject_anomalies(
        df,
        n=10,
        spike_ratio=0.5,
        seed=123,
        min_index=10,
        intensity="moderate",
    )

    # exatamente n labels == 1
    assert int(out["label"].sum()) == 10

    # nenhum injetado pode ser partial
    injected_rows = out[out["label"] == 1]
    assert (injected_rows["data_quality"] == "complete").all()

    # nenhum injetado pode estar antes de min_index
    injected_idx = injected_rows.index.to_numpy()
    assert (injected_idx >= 10).all()


def test_inject_anomalies_reproducible_seed():
    df = _make_df([10] * 60)

    out1 = inject_anomalies(df, n=12, seed=999, min_index=10, intensity="strong")
    out2 = inject_anomalies(df, n=12, seed=999, min_index=10, intensity="strong")

    # mesmas posições injetadas
    idx1 = out1.index[out1["label"] == 1].to_list()
    idx2 = out2.index[out2["label"] == 1].to_list()
    assert idx1 == idx2

    # mesmos tipos
    types1 = out1.loc[idx1, "injected_type"].to_list()
    types2 = out2.loc[idx2, "injected_type"].to_list()
    assert types1 == types2