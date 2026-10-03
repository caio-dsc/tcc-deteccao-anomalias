import pandas as pd


Z_SCORE_PATH = "data/processed/resultado_zscore.csv"
ISOLATION_PATH = "data/processed/resultado_isolation_forest.csv"
OUTPUT_PATH = "data/processed/comparacao_anomalias.csv"


def main() -> None:
    zscore = pd.read_csv(Z_SCORE_PATH)
    isolation = pd.read_csv(ISOLATION_PATH)

    columns_isolation = [
        "date",
        "isolation_score",
        "isolation_prediction",
        "anomaly_isolation_forest",
    ]

    comparison = zscore.merge(
        isolation[columns_isolation],
        on="date",
        how="inner",
    )

    comparison["agreement"] = "normal"

    comparison.loc[
        comparison["anomaly_zscore"]
        & comparison["anomaly_isolation_forest"],
        "agreement",
    ] = "both"

    comparison.loc[
        comparison["anomaly_zscore"]
        & ~comparison["anomaly_isolation_forest"],
        "agreement",
    ] = "zscore_only"

    comparison.loc[
        ~comparison["anomaly_zscore"]
        & comparison["anomaly_isolation_forest"],
        "agreement",
    ] = "isolation_only"

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total = len(comparison)
    zscore_count = comparison["anomaly_zscore"].sum()
    isolation_count = comparison["anomaly_isolation_forest"].sum()
    both_count = (
        comparison["agreement"] == "both"
    ).sum()
    zscore_only_count = (
        comparison["agreement"] == "zscore_only"
    ).sum()
    isolation_only_count = (
        comparison["agreement"] == "isolation_only"
    ).sum()

    print()
    print("=== COMPARAÇÃO DOS DETECTORES ===")
    print()
    print(f"Total de dias: {total}")
    print(f"Anomalias Z-score: {zscore_count}")
    print(
        f"Anomalias Isolation Forest: {isolation_count}"
    )
    print(f"Detectadas pelos dois: {both_count}")
    print(f"Somente Z-score: {zscore_only_count}")
    print(
        f"Somente Isolation Forest: "
        f"{isolation_only_count}"
    )

    print()
    print("=== ANOMALIAS DETECTADAS PELOS DOIS ===")

    both = comparison[
        comparison["agreement"] == "both"
    ]

    print(
        both[
            [
                "date",
                "consumption_kwh",
                "zscore",
                "isolation_score",
                "data_quality",
            ]
        ]
        .sort_values("date")
        .to_string(index=False)
    )

    print()
    print("=== SOMENTE Z-SCORE ===")

    zscore_only = comparison[
        comparison["agreement"] == "zscore_only"
    ]

    print(
        zscore_only[
            [
                "date",
                "consumption_kwh",
                "zscore",
                "data_quality",
            ]
        ]
        .sort_values("date")
        .to_string(index=False)
    )

    print()
    print("=== SOMENTE ISOLATION FOREST ===")

    isolation_only = comparison[
        comparison["agreement"] == "isolation_only"
    ]

    print(
        isolation_only[
            [
                "date",
                "consumption_kwh",
                "isolation_score",
                "data_quality",
            ]
        ]
        .sort_values("date")
        .to_string(index=False)
    )

    print()
    print(f"Resultado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
