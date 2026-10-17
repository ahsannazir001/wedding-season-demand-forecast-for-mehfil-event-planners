
"""Train and evaluate a feature-engineered wedding-bookings model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "wedding_bookings.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
PLOT_DIR = PROJECT_ROOT / "plots"

TEST_MONTHS = 8
PEAK_MONTHS = [6, 7, 8]


def load_data() -> pd.DataFrame:
    """Load data and create the required time-based features."""
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

    month_map = {
        name: number
        for number, name in enumerate(
            pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%B"),
            start=1,
        )
    }
    df["month_number"] = df["month"].map(month_map)

    if df["month_number"].isnull().any():
        raise ValueError("Dataset contains an invalid month name.")

    if df.duplicated(subset=["year", "month"]).any():
        raise ValueError("Duplicate year-month records found.")

    first_year = df["year"].min()
    df["time_index"] = (
        (df["year"] - first_year) * 12 + df["month_number"] - 1
    )
    df["year_offset"] = df["year"] - first_year
    df["is_peak_season"] = df["month_number"].isin(PEAK_MONTHS).astype(int)

    df = df.sort_values("time_index").reset_index(drop=True)

    if not df["bookings"].ge(0).all():
        raise ValueError("Booking counts cannot be negative.")

    return df


def main() -> None:
    df = load_data()

    features = [
        "time_index",
        "month_number",
        "is_peak_season",
        "year_offset",
    ]
    target = "bookings"

    train = df.iloc[:-TEST_MONTHS].copy()
    test = df.iloc[-TEST_MONTHS:].copy()

    model = LinearRegression()
    model.fit(train[features], train[target])

    predictions = model.predict(test[features]).clip(min=0)
    test_mae = mean_absolute_error(test[target], predictions)

    # Expanding-window validation preserves chronological order.
    cv = TimeSeriesSplit(n_splits=4)
    cv_model = LinearRegression()
    cv_scores = -cross_val_score(
        cv_model,
        train[features],
        train[target],
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    cv_mae = cv_scores.mean()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORT_DIR / "improved_metrics.txt"
    plot_file = PLOT_DIR / "improved_actual_vs_predicted.png"

    report = (
        "Wedding Season Demand Forecast - Feature Engineering\n"
        "Model: Linear Regression\n"
        f"Features: {', '.join(features)}\n"
        f"Training rows: {len(train)}\n"
        f"Testing rows: {len(test)}\n"
        f"Test period: {test.iloc[0]['month']} "
        f"{int(test.iloc[0]['year'])} to "
        f"{test.iloc[-1]['month']} {int(test.iloc[-1]['year'])}\n"
        f"Test MAE: {test_mae:.4f} bookings\n"
        f"TimeSeriesSplit folds: {len(cv_scores)}\n"
        f"Cross-validation MAE scores: "
        f"{', '.join(f'{score:.4f}' for score in cv_scores)}\n"
        f"Mean cross-validation MAE: {cv_mae:.4f} bookings\n"
        "Peak season assumption: June, July, August\n"
        "Note: Dataset is synthetic; results are for learning.\n"
    )

    print(report)
    report_file.write_text(report, encoding="utf-8")

    labels = [
        f"{row.month[:3]} {int(row.year)}"
        for row in test.itertuples()
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(labels, test[target].to_numpy(), marker="o", label="Actual")
    plt.plot(labels, predictions, marker="s", linestyle="--", label="Improved model")
    plt.xticks(rotation=45)
    plt.xlabel("Test month")
    plt.ylabel("Bookings")
    plt.title("Feature-Engineered Model: Actual vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Metrics saved to: {report_file}")
    print(f"Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
