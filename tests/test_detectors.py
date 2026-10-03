import pandas as pd

from src.detectors import detect_zscore_rolling


def test_zscore_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=10,
                freq="D",
            ),
            "consumption_kwh": [
                10,
                11,
                10,
                12,
                11,
                10,
                11,
                10,
                11,
                100,
            ],
            "data_quality": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
            ],
        }
    )

    result = detect_zscore_rolling(
        df,
        window=7,
        threshold=3.0,
    )

    assert "rolling_mean" in result.columns
    assert "rolling_std" in result.columns
    assert "zscore" in result.columns
    assert "anomaly_zscore" in result.columns


def test_zscore_detects_large_spike():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=10,
                freq="D",
            ),
            "consumption_kwh": [
                10,
                11,
                9,
                12,
                10,
                11,
                9,
                10,
                11,
                100,
            ],
            "data_quality": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
            ],
        }
    )

    result = detect_zscore_rolling(
        df,
        window=7,
        threshold=3.0,
    )

    assert result.iloc[-1]["anomaly_zscore"]


def test_partial_day_is_not_detected():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=10,
                freq="D",
            ),
            "consumption_kwh": [
                10,
                11,
                9,
                12,
                10,
                11,
                9,
                10,
                11,
                100,
            ],
            "data_quality": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "partial",
            ],
        }
    )

    result = detect_zscore_rolling(
        df,
        window=7,
        threshold=3.0,
    )

    assert not result.iloc[-1]["anomaly_zscore"]
