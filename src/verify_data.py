
"""Verify the generated wedding booking dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_ROOT / "data" / "processed" / "wedding_bookings.csv"


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    expected_columns = ["month", "year", "bookings", "budget"]

    assert list(df.columns) == expected_columns, "Incorrect columns."
    assert len(df) == 36, "Dataset must contain exactly 36 rows."
    assert not df.isnull().any().any(), "Missing values detected."
    assert not df.duplicated(subset=["year", "month"]).any(), (
        "Duplicate month-year records detected."
    )
    assert (df["bookings"] >= 0).all(), "Negative booking count found."
    assert (df["budget"] > 0).all(), "Invalid budget found."

    print("All dataset checks passed!")
    print(f"Total rows: {len(df)}")
    print(f"Unique years: {sorted(df['year'].unique().tolist())}")
    print(f"Booking range: {df['bookings'].min()} - {df['bookings'].max()}")
    print(f"Average budget: {df['budget'].mean():,.2f}")
    print("\nData types:")
    print(df.dtypes)
    print("\nLast five records:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
