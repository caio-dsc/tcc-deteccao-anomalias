import pandas as pd


def detect_zscore_rolling(
    df: pd.DataFrame,
    window: int = 7,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detecta possíveis anomalias usando Z-score
    calculado sobre uma janela móvel.

    Parâmetros:
        df:
            DataFrame contendo:
            - date
            - consumption_kwh
            - data_quality

        window:
            Quantidade de observações anteriores
            utilizadas na janela.

        threshold:
            Limite absoluto do Z-score para sinalização.

    Retorno:
        DataFrame contendo as colunas originais
        e as informações calculadas pelo detector.
    """

    result = df.copy()

    # Garante que os dados estejam ordenados cronologicamente.
    result["date"] = pd.to_datetime(result["date"])

    result = result.sort_values("date").reset_index(drop=True)

    # Inicialmente, apenas dias completos participam
    # da detecção.
    valid = result["data_quality"] == "complete"

    values = result["consumption_kwh"].where(valid)

    # Média móvel dos valores anteriores.
    rolling_mean = (
        values
        .shift(1)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    # Desvio-padrão móvel dos valores anteriores.
    rolling_std = (
        values
        .shift(1)
        .rolling(
            window=window,
            min_periods=window,
        )
        .std()
    )

    result["rolling_mean"] = rolling_mean
    result["rolling_std"] = rolling_std

    # Calcula o Z-score.
    result["zscore"] = (
        result["consumption_kwh"] - result["rolling_mean"]
    ) / result["rolling_std"]

    # Inicialmente todos os registros não são anomalias.
    result["anomaly_zscore"] = False

    # Só sinaliza quando:
    # 1. o dia possui dados completos;
    # 2. existe média e desvio-padrão calculados;
    # 3. o desvio-padrão é diferente de zero;
    # 4. o valor absoluto do Z-score ultrapassa o limite.
    valid_zscore = (
        valid
        & result["rolling_mean"].notna()
        & result["rolling_std"].notna()
        & (result["rolling_std"] > 0)
        & (result["zscore"].abs() >= threshold)
    )

    result.loc[
        valid_zscore,
        "anomaly_zscore",
    ] = True

    return result
