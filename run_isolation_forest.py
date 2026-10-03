import pandas as pd

from src.isolation_forest import detect_isolation_forest


INPUT_PATH = "data/processed/consumo_diario.csv"
OUTPUT_PATH = "data/processed/resultado_isolation_forest.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    result = detect_isolation_forest(
        df,
        contamination=0.025,
        n_estimators=200,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    anomalies = result[
        result["anomaly_isolation_forest"]
    ].copy()

    print()
    print("=== RESULTADO DO ISOLATION FOREST ===")
    print()
    print(f"Total de dias: {len(result)}")
    print(
        "Dias completos:",
        (result["data_quality"] == "complete").sum(),
    )
    print(
        "Dias parciais:",
        (result["data_quality"] != "complete").sum(),
    )
    print(
        "Anomalias detectadas:",
        len(anomalies),
    )

    print()
    print("Primeiras anomalias:")

    columns = [
        "date",
        "consumption_kwh",
        "isolation_score",
        "isolation_prediction",
    ]

    print(
        anomalies[columns]
        .sort_values(
            "isolation_score",
            ascending=True,
        )
        .head(10)
        .to_string(index=False)
    )

    print()
    print(f"Resultado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
