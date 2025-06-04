# SEM Budget Optimization Tool

A Python-based tool for optimizing Search Engine Marketing (SEM) budget allocation across different ad groups. This tool uses statistical modeling and optimization techniques to recommend optimal budget distribution while maximizing conversions.

## Features

- Interactive dashboard for budget optimization visualization
- Statistical modeling of conversion response curves
- Marginal return optimization
- Detailed business justifications for budget recommendations
- Export capabilities for results and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bkhatib/sem_budget_allocation.git
cd sem_budget_allocation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your SEM data in CSV format in the `sem_analysis` directory with the name `SEM_DATA_TOP100.csv`

2. Run the Streamlit dashboard:
```bash
streamlit run sem_analysis/app.py
```

3. Access the dashboard in your web browser (typically at http://localhost:8501)

## Dashboard Features

- Adjustable total budget through interactive controls
- Summary metrics showing key performance indicators
- Interactive budget allocation comparison charts
- TCPA comparison visualizations
- Detailed results table with formatted numbers
- Expandable business justifications for each ad group
- Download options for results and response curves

## Data Requirements

The input CSV file should contain the following columns:
- AdGroup: Name of the ad group
- Spend: Weekly spend amount
- Conversions: Number of conversions
- Impressions: Number of impressions
- Clicks: Number of clicks
- week_start: Start date of the week

## License

MIT License 