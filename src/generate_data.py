
"""Generate and clean synthetic monthly wedding booking data."""

from pathlib import Path

import numpy as np
import pandas as pd


# Project root is the parent directory of src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "wedding_bookings.csv"

RANDOM_SEED = 42
START_DATE = "2023-01-01"
MONTH_COUNT = 36


def generate_booking_data() -> pd.DataFrame:
    """Generate 36 months of synthetic wedding booking records."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, periods=MONTH_COUNT, freq="MS")

    records = []

    for date in dates:
        month = date.month

        # Illustrative seasonal demand, not real client observations.
        seasonal_effect = {
            1: 1.35, 2: 1.15, 3: 0.85, 4: 0.75,
            5: 0.90, 6: 0.70, 7: 0.65, 8: 0.80,
            9: 0.90, 10: 1.10, 11: 1.40, 12: 1.55,
        }[month]

        trend_effect = 1 + 0.04 * (date.year - 2023)
        expected_bookings = 20 * seasonal_effect * trend_effect

        bookings = max(
            0, int(round(rng.normal(expected_bookings, 3.0)))
        )

        average_budget = max(
            50000,
            int(round(rng.normal(250000 * trend_effect, 25000), -3)),
        )

        records.append(
            {
                "month": date.strftime("%B"),
                "year": date.year,
                "bookings": bookings,
                "budget": average_budget,
            }
        )

    data = pd.DataFrame(records)

    # Simulate missing values to demonstrate data cleaning.
    data.loc[[4, 17], "bookings"] = np.nan
    data.loc[[8, 25], "budget"] = np.nan

    return data


def clean_booking_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean and return the monthly booking dataset."""
    required_columns = ["month", "year", "bookings", "budget"]

    if list(data.columns) != required_columns:
        raise ValueError("Dataset columns do not match the required schema.")

    if len(data) != MONTH_COUNT:
        raise ValueError(f"Expected {MONTH_COUNT} monthly rows.")

    cleaned = data.copy()

    # Validate month names.
    valid_months = set(pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%B"))
    if not cleaned["month"].isin(valid_months).all():
        raise ValueError("Dataset contains invalid month names.")

    # Fill missing numeric values using the median.
    for column in ["bookings", "budget"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    # Ensure valid values and appropriate integer data types.
    if (cleaned["bookings"] < 0).any():
        raise ValueError("Bookings cannot be negative.")

    if (cleaned["budget"] <= 0).any():
        raise ValueError("Budget must be greater than zero.")

    if not np.equal(cleaned["bookings"], np.floor(cleaned["bookings"])).all():
        raise ValueError("Booking counts must be whole numbers.")

    if not np.equal(cleaned["budget"], np.floor(cleaned["budget"])).all():
        raise ValueError("Budget values must be whole numbers.")

    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="raise").astype("int64")
    cleaned["bookings"] = cleaned["bookings"].astype("int64")
    cleaned["budget"] = cleaned["budget"].astype("int64")

    # Ensure every month-year pair is unique.
    if cleaned.duplicated(subset=["year", "month"]).any():
        raise ValueError("Duplicate month-year records found.")

    # Sort chronologically.
    month_order = {
        month: index
        for index, month in enumerate(
            pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%B"),
            start=1,
        )
    }
    cleaned["_month_number"] = cleaned["month"].map(month_order)
    cleaned = cleaned.sort_values(["year", "_month_number"])
    cleaned = cleaned.drop(columns="_month_number").reset_index(drop=True)

    if cleaned.isnull().any().any():
        raise ValueError("Missing values remain after cleaning.")

    return cleaned


def main() -> None:
    """Generate, clean, save and verify the dataset."""
    raw_data = generate_booking_data()
    cleaned_data = clean_booking_data(raw_data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(OUTPUT_FILE, index=False)

    # Read the saved file back to verify the output.
    saved_data = pd.read_csv(OUTPUT_FILE)

    assert len(saved_data) == 36
    assert list(saved_data.columns) == ["month", "year", "bookings", "budget"]
    assert not saved_data.isnull().any().any()

    print("Dataset generated and cleaned successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(saved_data)}")
    print(f"Columns: {list(saved_data.columns)}")
    print(f"Missing values: {int(saved_data.isnull().sum().sum())}")
    print("\nFirst five records:")
    print(saved_data.head().to_string(index=False))


if __name__ == "__main__":
    main()
