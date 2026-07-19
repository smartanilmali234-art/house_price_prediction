import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data():
    data_path = "data/house_prices.csv"
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        features = data.drop(columns=["Price"])
        target = data["Price"]
        return features, target

    housing = fetch_california_housing(as_frame=True)
    features = housing.data
    target = housing.target
    return features, target


def train_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return model, y_test, predictions


def evaluate_model(y_test, predictions):
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions),
    }


def save_prediction_plot(y_test, predictions):
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual House Prices")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.tight_layout()
    plt.savefig("images/predicted_vs_actual.png", dpi=150)
    plt.close()


def main():
    features, target = load_data()
    _, y_test, predictions = train_model(features, target)
    metrics = evaluate_model(y_test, predictions)

    print("House Price Prediction Results")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    print("\nSample predictions:")
    print(results.head())

    save_prediction_plot(y_test, predictions)
    print("\nPlot saved to images/predicted_vs_actual.png")


if __name__ == "__main__":
    main()
