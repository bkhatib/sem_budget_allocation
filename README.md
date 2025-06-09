# SEM Budget Optimization Dashboard

A powerful dashboard for optimizing Search Engine Marketing (SEM) budget allocation across different ad groups using advanced statistical models.

## Features

- **Global Marginal Return Model**: Optimizes budget allocation using logarithmic regression
- **Multi-Factor Model**: Advanced optimization considering multiple variables (spend, CTR, CVR)
- **Interactive Dashboard**: Visualize and analyze optimization results
- **Detailed Metrics**: Comprehensive performance indicators and business justifications
- **Export Capabilities**: Download results and response curves

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sem_budget_allocation.git
cd sem_budget_allocation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
cd sem_analysis
streamlit run app.py
```

2. Upload your SEM data CSV file or use the default dataset
3. Select your preferred model type
4. Adjust the total budget as needed
5. View and analyze the optimization results

## Data Format

The input CSV file should contain the following columns:
- AdGroup: Name of the ad group
- week_start: Start date of the week
- Spend: Amount spent
- Conversions: Number of conversions
- Clicks: Number of clicks
- Impressions: Number of impressions

## Models

### Global Marginal Return Model
- Uses logarithmic function to model spend vs. conversions
- Optimizes based on marginal returns
- Best for basic budget allocation

### Multi-Factor Model
- Considers multiple variables affecting conversions
- More sophisticated optimization approach
- Better for complex scenarios

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 