from src.preprocessing import process_dataset


INPUT_PATH = (
    "data/raw/household_power_consumption.txt"
)

OUTPUT_PATH = (
    "data/processed/consumo_diario.csv"
)


if __name__ == "__main__":
    daily = process_dataset(
        INPUT_PATH,
        OUTPUT_PATH,
    )

    print("Processamento concluído!")
    print(f"Dias processados: {len(daily)}")
    print()
    print(daily.head())
