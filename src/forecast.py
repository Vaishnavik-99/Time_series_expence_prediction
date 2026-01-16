import pickle
import pandas as pd
from preprocess import load_and_preprocess


def forecast_expense(model_path, data_path, steps=6):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data = load_and_preprocess(data_path)

    forecast = model.forecast(steps=steps)

    future_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='M')[1:]
    forecast_df = pd.DataFrame({'date': future_dates, 'predicted_expense': forecast.values})

    return forecast_df


if __name__ == "__main__":
    result = forecast_expense("../models/sarima_model.pkl", "../data/expenses.csv")
    print(result)
