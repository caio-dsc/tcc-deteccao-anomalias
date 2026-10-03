from pathlib import Path

import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset bruto de consumo elétrico.

    O dataset utiliza ponto e vírgula como separador
    e '?' para representar valores ausentes.
    """

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para análise.

    - Converte Global_active_power para numérico.
    - Combina Date e Time em uma coluna datetime.
    - Remove registros sem data/hora ou potência válida.
    """

    df = df.copy()

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce",
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["datetime", "Global_active_power"]
    )

    return df


def calculate_daily_consumption(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula o consumo diário em kWh,
    quantidade de medições válidas,
    cobertura e qualidade dos dados.
    """

    df = df.copy()

    # Cada registro representa aproximadamente um minuto.
    # Global_active_power está em kW.
    # Portanto, kW / 60 = kWh por minuto.
    df["energy_kwh"] = (
        df["Global_active_power"] / 60.0
    )

    # Obtém somente a data.
    df["date"] = df["datetime"].dt.date

    # Consumo diário.
    consumption = (
        df.groupby("date")["energy_kwh"]
        .sum()
        .rename("consumption_kwh")
    )

    # Quantidade de medições válidas.
    valid_measurements = (
        df.groupby("date")["Global_active_power"]
        .count()
        .rename("valid_measurements")
    )

    # Junta as informações.
    daily = pd.concat(
        [
            consumption,
            valid_measurements,
        ],
        axis=1,
    ).reset_index()

    # Um dia completo possui 1.440 minutos.
    daily["coverage"] = (
        daily["valid_measurements"] / 1440.0
    )

    # Classificação inicial da qualidade.
    daily["data_quality"] = daily[
        "valid_measurements"
    ].apply(
        lambda x: (
            "complete"
            if x == 1440
            else "partial"
        )
    )

    return daily


def process_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Executa o pipeline completo de processamento
    e salva o resultado em CSV.
    """

    df = load_raw_data(input_path)

    df = prepare_data(df)

    daily = calculate_daily_consumption(df)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        output_path,
        index=False,
    )

    return daily
