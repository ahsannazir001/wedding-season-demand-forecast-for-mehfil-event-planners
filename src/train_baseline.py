
"""Train and evaluate a baseline wedding-bookings forecasting model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a GUI.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "wedding_bookings.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
PLOT_DIR = PROJECT_ROOT / "plots"

TEST_MONTHS = 8


def load_data() -> pd.DataFrame:
    """Load and validate the cleaned booking dataset."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required = {"month", "year", "bookings"}

    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    if len(df) != 36:
        raise ValueError(f"Expected 36 rows, found {len(df)}")

    if df[["month", "year", "bookings"]].isnull().any().any():
        raise ValueError("Required columns contain missing values.")

    df["year"] = pd.to_numeric(df["year"], errors="raise")
    df["bookings"] = pd.to_numeric(df["bookings"], errors="raise")

    month_numbers = {
        month: number
        for number, month in enumerate(
            pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%B"),
            start=1,
        )
    }

    df["month_number"] = df["month"].map(month_numbers)

    if df["month_number"].isnull().any():
        raise ValueError("Dataset contains an invalid month name.")

    if df.duplicated(subset=["year", "month"]).any():
        raise ValueError("Duplicate year-month records found.")

    df["time_index"] = (
        (df["year"] - df["year"].min()) * 12 + df["month_number"] - 1
    )

    df = df.sort_values("time_index").reset_index(drop=True)

    if not df["bookings"].ge(0).all():
        raise ValueError("Booking counts cannot be negative.")

    return df


def main() -> None:
    """Train, evaluate, and save baseline forecast outputs."""
    df = load_data()

    # Preserve chronological order: latest months are held out for testing.
    train = df.iloc[:-TEST_MONTHS].copy()
    test = df.iloc[-TEST_MONTHS:].copy()

    features = ["time_index"]
    target = "bookings"

    model = LinearRegression()
    model.fit(train[features], train[target])

    predictions = model.predict(test[features])
    predictions = predictions.clip(min=0)

    mae = mean_absolute_error(test[target], predictions)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "baseline_metrics.txt"
    plot_file = PLOT_DIR / "actual_vs_predicted.png"

    report = (
        "Wedding Season Demand Forecast — Baseline Evaluation\n"
        "Model: Linear Regression\n"
        "Feature: time_index\n"
        f"Training rows: {len(train)}\n"
        f"Testing rows: {len(test)}\n"
        f"Test period: {test.iloc[0]['month']} {int(test.iloc[0]['year'])}"
        f" to {test.iloc[-1]['month']} {int(test.iloc[-1]['year'])}\n"
        f"Mean Absolute Error (MAE): {mae:.4f} bookings\n"
        "\nNote: The dataset is synthetic; this result is a learning baseline.\n"
    )

    print(report)
    report_file.write_text(report, encoding="utf-8")

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(test)),
        test[target].to_numpy(),
        marker="o",
        label="Actual bookings",
    )
    plt.plot(
        range(len(test)),
        predictions,
        marker="s",
        linestyle="--",
        label="Predicted bookings",
    )
    labels = [
        f"{row.month[:3]} {int(row.year)}"
        for row in test.itertuples()
    ]
    plt.xticks(range(len(test)), labels, rotation=45)
    plt.xlabel("Test month")
    plt.ylabel("Number of bookings")
    plt.title("Actual vs. Predicted Wedding Bookings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Metrics saved to: {report_file}")
    print(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
