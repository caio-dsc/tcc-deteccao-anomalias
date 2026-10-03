import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.025,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detecta anomalias no consumo diário utilizando Isolation Forest.

    Parâmetros:
        df:
            DataFrame contendo a coluna consumption_kwh.

        contamination:
            Proporção esperada de anomalias.

        n_estimators:
            Número de árvores utilizadas pelo modelo.

        random_state:
            Semente para garantir resultados reproduzíveis.

    Retorna:
        DataFrame original acrescido das colunas:

        isolation_score:
            Score de anormalidade calculado pelo modelo.

        isolation_prediction:
            1 para observação normal e -1 para anomalia.

        anomaly_isolation_forest:
            True quando a observação é classificada como anomalia.
    """

    if "consumption_kwh" not in df.columns:
        raise ValueError(
            "O DataFrame precisa conter a coluna 'consumption_kwh'."
        )

    result = df.copy()

    X = result[["consumption_kwh"]].copy()

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    model.fit(X)

    result["isolation_score"] = model.decision_function(X)

    result["isolation_prediction"] = model.predict(X)

    result["anomaly_isolation_forest"] = (
        result["isolation_prediction"] == -1
    )

    return result
