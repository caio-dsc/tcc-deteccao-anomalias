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

# =========================
# IQR (Tukey) detector
# =========================
def detect_iqr(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """
    Detecta possíveis anomalias usando o método IQR (Tukey).

    Regras:
    - Calcula Q1/Q3/IQR somente usando dias com data_quality == "complete".
    - Um dia é sinalizado como anomalia quando consumption_kwh < lower ou > upper.
    - Dias "partial" não são sinalizados.
    """
    result = df.copy()

    # Ordenação e tipo da data
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    # Apenas dias completos entram no cálculo dos quartis
    valid = result["data_quality"] == "complete"
    complete_values = result.loc[valid, "consumption_kwh"]

    # Cria colunas padrão (caso dataset pequeno não permita cálculo)
    result["iqr_q1"] = pd.NA
    result["iqr_q3"] = pd.NA
    result["iqr_value"] = pd.NA
    result["iqr_lower"] = pd.NA
    result["iqr_upper"] = pd.NA
    result["anomaly_iqr"] = False

    # Se não houver dias completos, não há como calcular IQR
    if complete_values.empty:
        return result

    q1 = complete_values.quantile(0.25)
    q3 = complete_values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    result["iqr_q1"] = q1
    result["iqr_q3"] = q3
    result["iqr_value"] = iqr
    result["iqr_lower"] = lower
    result["iqr_upper"] = upper

    out_of_bounds = (
        (result["consumption_kwh"] < lower) |
        (result["consumption_kwh"] > upper)
    )

    result.loc[valid & out_of_bounds, "anomaly_iqr"] = True

    return result

    # =========================
# Moving average detector
# =========================
def detect_moving_average(
    df: pd.DataFrame,
    window: int = 30,
    rel_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Detecta possíveis anomalias comparando o consumo do dia
    com a média móvel dos dias anteriores.

    Regras:
    - A média móvel é calculada com shift(1) para usar apenas dias anteriores.
    - Apenas dias com data_quality == "complete" participam do cálculo e podem ser sinalizados.
    - Anomalia quando:
        abs(consumo - media_movel) / media_movel >= rel_threshold
    """
    result = df.copy()

    # Ordenação e tipo da data
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    # Apenas dias completos entram no cálculo
    valid = result["data_quality"] == "complete"
    values = result["consumption_kwh"].where(valid)

    # Média móvel usando somente o passado
    ma_mean = (
        values
        .shift(1)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    result["ma_mean"] = ma_mean

    # Desvio relativo
    rel_dev = (result["consumption_kwh"] - result["ma_mean"]).abs() / result["ma_mean"]
    result["ma_rel_dev"] = rel_dev

    # Inicialmente, ninguém é anomalia
    result["anomaly_ma"] = False

    valid_ma = (
        valid
        & result["ma_mean"].notna()
        & (result["ma_mean"] > 0)
        & (result["ma_rel_dev"] >= rel_threshold)
    )

    result.loc[valid_ma, "anomaly_ma"] = True

    return result