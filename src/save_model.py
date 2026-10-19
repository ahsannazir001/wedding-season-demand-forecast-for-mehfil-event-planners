
"""Train the improved model on all available data and save it."""

from pathlib import Path
import joblib

from sklearn.linear_model import LinearRegression

from train_improved import load_data, PEAK_MONTHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "improved_model.joblib"

FEATURES = [
    "time_index",
    "month_number",
    "is_peak_season",
    "year_offset",
]


def main():
    df = load_data()

    # Final deployment model: train on all available historical data.
    model = LinearRegression()
    model.fit(df[FEATURES], df["bookings"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "first_year": int(df["year"].min()),
            "last_time_index": int(df["time_index"].max()),
            "peak_months": PEAK_MONTHS,
        },
        MODEL_FILE,
    )

    print("Improved model trained on all available data.")
    print(f"Training rows: {len(df)}")
    print(f"Saved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
