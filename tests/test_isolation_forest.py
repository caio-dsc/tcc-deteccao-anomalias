import pandas as pd

from src.isolation_forest import detect_isolation_forest


def test_isolation_forest_creates_expected_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=20,
                freq="D",
            ),
            "consumption_kwh": [
                20,
                21,
                19,
                20,
                22,
                21,
                20,
                19,
                21,
                20,
                22,
                19,
                20,
                21,
                20,
                22,
                19,
                21,
                20,
                100,
            ],
        }
    )

    result = detect_isolation_forest(
        df,
        contamination=0.05,
        random_state=42,
    )

    assert "isolation_score" in result.columns
    assert "isolation_prediction" in result.columns
    assert "anomaly_isolation_forest" in result.columns


def test_isolation_forest_detects_large_value():
    df = pd.DataFrame(
        {
            "consumption_kwh": [
                20,
                21,
                19,
                20,
                22,
                21,
                20,
                19,
                21,
                20,
                22,
                19,
                20,
                21,
                20,
                22,
                19,
                21,
                20,
                100,
            ]
        }
    )

    result = detect_isolation_forest(
        df,
        contamination=0.05,
        random_state=42,
    )

    assert result.iloc[-1]["anomaly_isolation_forest"]


def test_isolation_forest_is_reproducible():
    df = pd.DataFrame(
        {
            "consumption_kwh": [
                10,
                11,
                10,
                12,
                11,
                10,
                100,
            ]
        }
    )

    result_1 = detect_isolation_forest(
        df,
        contamination=0.15,
        random_state=42,
    )

    result_2 = detect_isolation_forest(
        df,
        contamination=0.15,
        random_state=42,
    )

    assert result_1["isolation_prediction"].tolist() == (
        result_2["isolation_prediction"].tolist()
    )
