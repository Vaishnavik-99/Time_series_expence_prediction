import pandas as pd

def load_and_preprocess(csv_path):
    data = pd.read_csv(csv_path)

    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Sort by date
    data = data.sort_index()

    # Handle missing values
    data = data.fillna(method='ffill')

    return data


if __name__ == "__main__":
    df = load_and_preprocess("../data/expenses.csv")
    print(df.head())
