import pandas as pd

from src.detectors import (
    detect_zscore_rolling,
    detect_iqr,
    detect_moving_average,
)


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


# =========================
# Z-score rolling tests
# =========================

def test_zscore_creates_expected_columns():
    df = _make_df([10, 11, 9, 10, 11, 9, 10, 30])
    out = detect_zscore_rolling(df, window=3, threshold=3.0)

    assert "rolling_mean" in out.columns
    assert "rolling_std" in out.columns
    assert "zscore" in out.columns
    assert "anomaly_zscore" in out.columns


def test_zscore_detects_spike_on_last_day():
    # IMPORTANTE: histórico com variância para rolling_std > 0
    df = _make_df([10, 11, 9, 10, 11, 9, 10, 100])
    out = detect_zscore_rolling(df, window=3, threshold=3.0)

    assert out.iloc[-1]["anomaly_zscore"] == True


def test_zscore_partial_day_is_not_detected():
    # Último dia é partial e tem pico -> NÃO pode sinalizar
    df = _make_df(
        [10, 11, 9, 10, 11, 9, 10, 100],
        qualities=["complete"] * 7 + ["partial"],
    )
    out = detect_zscore_rolling(df, window=3, threshold=3.0)

    assert out.iloc[-1]["anomaly_zscore"] == False


def test_zscore_constant_series_no_anomaly():
    # Série constante -> rolling_std = 0 -> não deve sinalizar anomalia
    df = _make_df([10] * 12)
    out = detect_zscore_rolling(df, window=5, threshold=3.0)

    assert out["anomaly_zscore"].any() == False


# =========================
# IQR tests
# =========================

def test_iqr_creates_expected_columns():
    df = _make_df([10, 10, 10, 10, 10, 10, 10, 10, 10, 100])
    out = detect_iqr(df, k=1.5)

    assert "iqr_q1" in out.columns
    assert "iqr_q3" in out.columns
    assert "iqr_value" in out.columns
    assert "iqr_lower" in out.columns
    assert "iqr_upper" in out.columns
    assert "anomaly_iqr" in out.columns


def test_iqr_detects_spike():
    df = _make_df([10, 10, 10, 10, 10, 10, 10, 10, 10, 100])
    out = detect_iqr(df, k=1.5)

    assert out.iloc[-1]["anomaly_iqr"] == True


def test_iqr_detects_drop():
    df = _make_df([10, 10, 10, 10, 10, 10, 10, 10, 10, 1])
    out = detect_iqr(df, k=1.5)

    assert out.iloc[-1]["anomaly_iqr"] == True


def test_iqr_partial_day_not_detected():
    df = _make_df(
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 100],
        qualities=["complete"] * 9 + ["partial"],
    )
    out = detect_iqr(df, k=1.5)

    assert out.iloc[-1]["anomaly_iqr"] == False


# =========================
# Moving average tests
# =========================

def test_moving_average_creates_expected_columns():
    df = _make_df([10, 10, 10, 10, 10, 10, 10, 30])
    out = detect_moving_average(df, window=7, rel_threshold=0.5)

    assert "ma_mean" in out.columns
    assert "ma_rel_dev" in out.columns
    assert "anomaly_ma" in out.columns


def test_moving_average_detects_spike_on_last_day():
    # 7 dias de 10 (histórico) + 1 dia de pico
    df = _make_df([10, 10, 10, 10, 10, 10, 10, 30])
    out = detect_moving_average(df, window=7, rel_threshold=0.5)

    assert out.iloc[-1]["anomaly_ma"] == True


def test_moving_average_no_signal_when_not_enough_history():
    # window=7, só 5 dias -> ma_mean NaN e sem anomalias
    df = _make_df([10, 10, 10, 10, 30])
    out = detect_moving_average(df, window=7, rel_threshold=0.5)

    assert out["anomaly_ma"].any() == False


def test_moving_average_partial_day_not_detected():
    # Último dia partial com pico -> não pode sinalizar
    df = _make_df(
        [10, 10, 10, 10, 10, 10, 10, 30],
        qualities=["complete"] * 7 + ["partial"],
    )
    out = detect_moving_average(df, window=7, rel_threshold=0.5)

    assert out.iloc[-1]["anomaly_ma"] == False