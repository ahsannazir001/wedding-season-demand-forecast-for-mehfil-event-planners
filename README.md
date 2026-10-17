
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


## Task 2: Baseline Forecast

### Model

A Linear Regression model from scikit-learn was trained to predict
monthly wedding bookings using a chronological time index.

### Train/Test Split

- Training records: 28
- Testing records: 8
- Test period: May 2025 to December 2025

The latest 8 months were reserved for testing to preserve chronological
order.

### Evaluation

The model was evaluated using Mean Absolute Error (MAE).

- **MAE: 6.4407 bookings**

MAE represents the average absolute difference between actual and
predicted booking counts on the test data.

### Outputs

- `reports/baseline_metrics.txt` — evaluation metrics and test period.
- `plots/actual_vs_predicted.png` — actual versus predicted bookings plot.

### Limitations

The dataset contains synthetic sample data, and the baseline model uses
only a linear time trend. The evaluation is intended as a learning
benchmark, not a guarantee of real-world forecasting accuracy.
## Task 3: Improve Model with Feature Engineering

### Objective
The goal of this task was to improve the baseline forecasting model by adding time-based features and evaluating the model using chronological cross-validation.

### Features Added
- `month_number`: Represents months from 1 to 12.
- `is_peak_season`: Indicates whether a month is part of the assumed summer peak season (June, July, and August).
- `year_offset`: Represents the number of years since the first year in the dataset.
- `time_index`: Preserved from the baseline model to represent chronological order.

### Model and Evaluation
A Linear Regression model was retrained using the engineered features. The latest 8 months (May–December 2025) were kept as the test set.

| Metric | Baseline | Improved Model |
|---|---:|---:|
| Test MAE | 6.4407 | 3.9635 |

The improved model reduced test MAE by approximately 38.5%.

### Cross-Validation
TimeSeriesSplit with 4 folds was used on the training data.

- Fold MAE scores: 20.2000, 6.7628, 4.7975, 7.8427
- Mean cross-validation MAE: 9.9007 bookings

Cross-validation results vary across time periods, so further evaluation is needed.

### Output Files
- `src/train_improved.py` — Feature engineering, model training, and evaluation.
- `reports/improved_metrics.txt` — Test and cross-validation results.
- `plots/improved_actual_vs_predicted.png` — Actual versus predicted bookings.

### Limitations
The dataset is synthetic, and the peak-season months are an assumption. The results are for learning purposes and do not guarantee real-world forecasting accuracy.