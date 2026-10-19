
"""Flask web app for wedding booking demand forecasts."""
from datetime import date
from calendar import month_name
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "improved_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "wedding_bookings.csv"

app = Flask(__name__)


def load_forecast_data():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    first_year = saved["first_year"]
    peak_months = saved["peak_months"]

    history = pd.read_csv(DATA_PATH)
    history["year"] = pd.to_numeric(history["year"], errors="raise")
    history["bookings"] = pd.to_numeric(history["bookings"], errors="raise")

    month_map = {month_name[i]: i for i in range(1, 13)}
    history["month_number"] = history["month"].map(month_map)

    if history["month_number"].isnull().any():
        raise ValueError("Dataset contains an invalid month name.")

    today = date.today()
    current_year = today.year
    current_month = today.month

    future_rows = []

    for step in range(1, 7):
        month_index = current_month - 1 + step
        month_number = month_index % 12 + 1
        year = current_year + month_index // 12
        time_index = (year - first_year) * 12 + month_number - 1

        future_rows.append({
            "year": year,
            "month": month_name[month_number],
            "time_index": time_index,
            "month_number": month_number,
            "is_peak_season": int(month_number in peak_months),
            "year_offset": year - first_year,
        })

    future_df = pd.DataFrame(future_rows)
    predictions = model.predict(future_df[features]).clip(min=0)
    future_df["predicted_bookings"] = [
        int(round(value)) for value in predictions
    ]

    return future_df.to_dict(orient="records")

    predictions = model.predict(future_df[features]).clip(min=0)

    future_df["predicted_bookings"] = [
        int(round(value)) for value in predictions
    ]

    return future_df.to_dict(orient="records")


@app.route("/")
def index():
    forecasts = load_forecast_data()
    return render_template("index.html", forecasts=forecasts)


if __name__ == "__main__":
    app.run(debug=True)
