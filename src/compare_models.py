
"""Compare baseline and feature-engineered models fairly."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "wedding_bookings.csv"

TEST_MONTHS = 8


def main():
    df = pd.read_csv(DATA_FILE)

    month_map = {
        name: number
        for number, name in enumerate(
            pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%B"),
            start=1,
        )
    }

    df["year"] = pd.to_numeric(df["year"], errors="raise")
    df["bookings"] = pd.to_numeric(df["bookings"], errors="raise")
    df["month_number"] = df["month"].map(month_map)

    first_year = df["year"].min()
    df["time_index"] = (
        (df["year"] - first_year) * 12 + df["month_number"] - 1
    )
    df["year_offset"] = df["year"] - first_year
    df["is_peak_season"] = df["month_number"].isin([6, 7, 8]).astype(int)
    df = df.sort_values("time_index").reset_index(drop=True)

    train = df.iloc[:-TEST_MONTHS].copy()
    test = df.iloc[-TEST_MONTHS:].copy()

    target = "bookings"
    baseline_features = ["time_index"]
    improved_features = [
        "time_index",
        "month_number",
        "is_peak_season",
        "year_offset",
    ]

    results = []

    for name, features in [
        ("Baseline", baseline_features),
        ("Improved", improved_features),
    ]:
        model = LinearRegression()
        model.fit(train[features], train[target])
        predictions = model.predict(test[features]).clip(min=0)
        test_mae = mean_absolute_error(test[target], predictions)

        cv = TimeSeriesSplit(n_splits=4)
        cv_scores = -cross_val_score(
            LinearRegression(),
            train[features],
            train[target],
            cv=cv,
            scoring="neg_mean_absolute_error",
        )

        results.append({
            "Model": name,
            "Test MAE": test_mae,
            "Mean CV MAE": cv_scores.mean(),
            "CV fold scores": ", ".join(f"{x:.4f}" for x in cv_scores),
        })

    result_df = pd.DataFrame(results)
    print("\nModel comparison (same test period and CV folds):")
    print(result_df.to_string(index=False, formatters={
        "Test MAE": "{:.4f}".format,
        "Mean CV MAE": "{:.4f}".format,
    }))

    output = PROJECT_ROOT / "reports" / "model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(f"\nComparison saved to: {output}")


if __name__ == "__main__":
    main()
