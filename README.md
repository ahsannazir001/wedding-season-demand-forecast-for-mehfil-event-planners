
# Wedding Season Demand Forecast for Mehfil Event Planners

## Project Overview

This project generates and cleans synthetic monthly wedding booking data
for Mehfil Event Planners. The dataset will be used in later tasks to build
and evaluate a machine learning model for forecasting monthly wedding demand.

**Important:** The dataset contains synthetic sample data. It does not
represent actual booking records from Mehfil Event Planners.

## Task 1 Objectives

- Generate 36 months of monthly wedding booking data.
- Include month, year, bookings, and average event budget.
- Simulate missing values to demonstrate data cleaning.
- Fill missing numeric values using the median.
- Validate data types and basic data quality.
- Save the cleaned dataset as a CSV file.

## Project Structure

```text
.
├── data/
│   └── processed/
│       └── wedding_bookings.csv
├── src/
│   ├── generate_data.py
│   └── verify_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `wedding/` directory contains the local virtual environment and is
excluded from Git.

## Requirements

- Python 3.10
- pandas 2.2.3
- NumPy 2.2.6

## Setup

Create and activate the virtual environment on Windows PowerShell:

```powershell
python -m venv wedding
.\wedding\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Generate and Clean the Dataset

Run the following command from the project root:

```powershell
python src/generate_data.py
```

The cleaned dataset will be saved to:

`data/processed/wedding_bookings.csv`

## Verify the Dataset

Run:

```powershell
python src/verify_data.py
```

The verification script checks the row count, required columns, missing
values, duplicate month-year records, and valid booking and budget values.

## Data Cleaning Method

1. Generate synthetic records for 36 consecutive months.
2. Introduce missing values in the bookings and budget columns.
3. Fill missing numeric values using the median of the respective column.
4. Convert year, bookings, and budget to integer data types.
5. Validate non-negative booking counts and positive budgets.
6. Check for duplicate year-month records and remaining missing values.
7. Sort records chronologically and export the cleaned CSV.

## Dataset Schema

| Column | Description |
|---|---|
| month | Name of the month |
| year | Calendar year |
| bookings | Number of wedding bookings |
| budget | Average budget per event, in illustrative PKR |

## Reproducibility

A fixed random seed is used so that generated sample data can be reproduced
with the same code and compatible dependency versions.

## Limitations

This dataset is synthetic and is intended for learning and demonstration.
Its seasonal patterns and budget values are illustrative assumptions, not
verified business data. Forecast quality must be evaluated separately
during subsequent tasks.

## Task 1 Deliverables

- Python data generation and cleaning script
- Cleaned CSV containing 36 monthly records
- Dataset verification script
- README documentation
