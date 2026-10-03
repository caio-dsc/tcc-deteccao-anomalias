import pandas as pd

from src.detectors import detect_zscore_rolling


INPUT_PATH = "data/processed/consumo_diario.csv"
OUTPUT_PATH = "data/processed/resultado_zscore.csv"


def main():
    # Carrega os dados processados.
    df = pd.read_csv(INPUT_PATH)

    # Executa o detector.
    result = detect_zscore_rolling(
        df,
        window=7,
        threshold=3.0,
    )

    # Salva o resultado completo.
    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Seleciona somente as anomalias.
    anomalies = result[
        result["anomaly_zscore"]
    ].copy()

    print()
    print("=== RESULTADO DO Z-SCORE ===")
    print()
    print(f"Total de dias: {len(result)}")
    print(
        "Dias completos:",
        (result["data_quality"] == "complete").sum(),
    )
    print(
        "Dias parciais:",
        (result["data_quality"] == "partial").sum(),
    )
    print(
        "Anomalias detectadas:",
        len(anomalies),
    )

    print()

    if len(anomalies) > 0:
        print("Primeiras anomalias:")
        print(
            anomalies[
                [
                    "date",
                    "consumption_kwh",
                    "rolling_mean",
                    "rolling_std",
                    "zscore",
                ]
            ]
            .sort_values("zscore", key=lambda x: x.abs(), ascending=False)
            .head(10)
            .to_string(index=False)
        )
    else:
        print("Nenhuma anomalia foi detectada.")

    print()
    print(f"Resultado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
