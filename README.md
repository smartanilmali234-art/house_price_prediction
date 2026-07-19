# House Price Prediction

This project trains a machine learning model to predict house prices using a local sample dataset. If the CSV file is missing, the script falls back to the California Housing dataset from scikit-learn.

## Project Structure

```text
house-price-prediction/
|-- house_price_prediction.ipynb
|-- house_price_prediction.py
|-- README.md
|-- requirements.txt
|-- data/
|   `-- house_prices.csv
|-- images/
|   `-- predicted_vs_actual.png
```

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run

Run the Python script:

```bash
python house_price_prediction.py
```

The script trains a random forest regression model, prints evaluation metrics, and saves a predicted-vs-actual plot to `images/predicted_vs_actual.png`.

## Dataset

The local dataset is stored at `data/house_prices.csv`.

Target column:

- `Price`

Feature columns:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`
