
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
### Task 3: Feature Engineering and Model Comparison

**Objective:** Improve the baseline wedding-booking demand forecasting model by adding calendar-based features and evaluating performance using time-series cross-validation.

#### Features Added

- `time_index`: Number of months since the first observation, representing the passage of time.
- `month_number`: Month number from 1 to 12.
- `is_peak_season`: Binary indicator for June, July, and August. These months are assumed to represent the peak summer season for this learning project.
- `year_offset`: Number of years since the first year in the dataset.

#### Evaluation Method

Both models use the same chronological split: the first 28 observations for training and the final 8 observations (May–December 2025) for testing. Four-fold `TimeSeriesSplit` cross-validation is also used on the training data.

#### Model Comparison

| Metric | Baseline Model | Improved Model |
|---|---:|---:|
| Features | `time_index` | `time_index`, `month_number`, `is_peak_season`, `year_offset` |
| Test MAE | 6.4407 | 3.9635 |
| Mean Cross-Validation MAE | 9.6458 | 9.9007 |

The improved model's test MAE is approximately **38.5% lower** than the baseline model's test MAE on the same test period.

However, the improved model's mean cross-validation MAE is slightly higher than the baseline's. This means the improvement on the final test period does not establish that the improved model will generalize better across different time periods.

#### Generated Outputs

- `src/train_improved.py`: Feature-engineered model training and evaluation.
- `src/compare_models.py`: Fair comparison of baseline and improved models using the same chronological test period and cross-validation splits.
- `reports/improved_metrics.txt`: Improved model metrics.
- `reports/model_comparison.csv`: Side-by-side model comparison and cross-validation results.
- `plots/improved_actual_vs_predicted.png`: Plot comparing actual and predicted bookings for the improved model.

#### Limitations

The dataset is synthetic and contains only 36 monthly observations. The peak-season months are an assumption, not a confirmed business rule. The results are intended for learning and should be validated on real booking data before practical business use.
## Task 4: Deploy a Web App for Wedding Demand Forecasts

### Project Overview

This task extends the wedding booking demand forecasting project by developing a web application using Flask. The application loads the trained feature-engineered Linear Regression model and generates forecasts for the next six months for Mehfil Event Planners.

### Features

- **Six-Month Forecasting:** Displays predicted wedding bookings for the next six months.
- **Trained ML Model:** Uses the improved Linear Regression model trained on all available historical data.
- **Feature Engineering:** Uses `time_index`, `month_number`, `is_peak_season`, and `year_offset`.
- **Web Interface:** Displays forecast results in a simple, responsive HTML table.
- **Automatic Forecast Dates:** Calculates the next six months based on the current date.
- **Non-Negative Predictions:** Prevents negative forecast values and rounds predictions to whole bookings.

### Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML and CSS
- Gunicorn (for production deployment)

### Project Structure

```text
wedding-season-demand-forecast-for-mehfil-event-planners/
├── app.py
├── requirements.txt
├── models/
│   └── improved_model.joblib
├── data/
│   └── processed/
│       └── wedding_bookings.csv
├── src/
│   ├── train_improved.py
│   └── save_model.py
└── templates/
    └── index.html
```

### Run the Application Locally

**1. Clone the repository**

```bash
git clone https://github.com/ahsannazir001/wedding-season-demand-forecast-for-mehfil-event-planners.git
```

**2. Navigate to the project directory**

```bash
cd wedding-season-demand-forecast-for-mehfil-event-planners
```

**3. Create and activate a virtual environment**

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**5. Start the Flask application**

```bash
python app.py
```

**6. Open the application**

**Live Application:** [Wedding Demand Forecast — Mehfil Event Planners](https://wedding-season-demand-forecast-for.onrender.com/)

The application is deployed on Render and displays predicted wedding bookings for the next six months.

```