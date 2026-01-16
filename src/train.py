import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from preprocess import load_and_preprocess


def train_model(data_path, model_path):
    data = load_and_preprocess(data_path)

    model = SARIMAX(
        data['expense'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    model_fit = model.fit()

    with open(model_path, 'wb') as f:
        pickle.dump(model_fit, f)

    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_model("../data/expenses.csv", "../models/sarima_model.pkl")
