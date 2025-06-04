import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import matplotlib.pyplot as plt
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEEKLY_BUDGET = 40000
DATA_PATH = 'sem_analysis/sem_data_groups.csv'

# --- Helper Functions ---
def fit_saturation_model(group_df):
    """
    Fit a log-linear (diminishing returns) model: Conversions = a + b * log(Spend)
    Returns: model, a, b, R^2
    """
    df = group_df[group_df['Spend'] > 0].copy()
    if len(df) < 2:
        return None, np.nan, np.nan, np.nan
    X = np.log(df['Spend']).values.reshape(-1, 1)
    X = sm.add_constant(X)
    y = df['Conversions'].values
    model = sm.OLS(y, X).fit()
    a, b = model.params
    return model, a, b, model.rsquared

def compute_efficiency_metrics(group_df):
    """
    Compute conversion rate, revenue per click, and impression-to-conversion rate.
    """
    clicks = group_df['Clicks'].sum()
    impressions = group_df['Impressions'].sum()
    conversions = group_df['Conversions'].sum()
    revenue = group_df['Revenue'].sum()
    spend = group_df['Spend'].sum()
    cvr = conversions / clicks if clicks > 0 else 0
    rpc = revenue / clicks if clicks > 0 else 0
    imp_cvr = conversions / impressions if impressions > 0 else 0
    tcpas = spend / conversions if conversions > 0 else np.nan
    return cvr, rpc, imp_cvr, tcpas

def optimal_spend_for_adgroup(a, b, cvr, rpc, tcpas, weekly_budget, min_spend, max_spend):
    """
    Find the spend that maximizes conversions under diminishing returns, subject to constraints.
    Uses the log-linear model: conversions = a + b * log(spend)
    """
    # Marginal CPA = TCPA at optimal point
    # d(conversions)/d(spend) = b / spend
    # TCPA = spend / conversions
    # We want to maximize conversions, but not let TCPA exceed current or target
    # We'll use a grid search for simplicity
    spends = np.linspace(min_spend, max_spend, 100)
    conversions = a + b * np.log(spends)
    tcpas_grid = spends / conversions
    # Only consider spends where TCPA is not much higher than current
    mask = (conversions > 0) & (tcpas_grid <= tcpas * 1.2)  # allow up to 20% higher than historical
    if not np.any(mask):
        idx = np.argmax(conversions)
    else:
        idx = np.argmax(conversions[mask])
    return spends[idx], tcpas_grid[idx], conversions[idx]

# --- Main MMM Optimization ---
def mmm_optimize_weekly_budget(df, weekly_budget=WEEKLY_BUDGET):
    logger.info(f"Optimizing weekly budget allocation for total ${weekly_budget:,.0f}...")
    adgroup_results = []
    for adgroup, group_df in df.groupby('AdGroup'):
        logger.info(f"Fitting model for {adgroup}...")
        model, a, b, r2 = fit_saturation_model(group_df)
        cvr, rpc, imp_cvr, tcpas = compute_efficiency_metrics(group_df)
        min_spend = max(10, group_df['Spend'].min())
        max_spend = max(100, group_df['Spend'].max() * 1.5)
        if np.isnan(a) or np.isnan(b) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            continue
        opt_spend, opt_tcpa, opt_convs = optimal_spend_for_adgroup(a, b, cvr, rpc, tcpas, weekly_budget, min_spend, max_spend)
        adgroup_results.append({
            'AdGroup': adgroup,
            'Model_a': a,
            'Model_b': b,
            'R2': r2,
            'Current_CVR': cvr,
            'Revenue_per_Click': rpc,
            'Impression_CVR': imp_cvr,
            'Current_TCPA': tcpas,
            'Recommended_Weekly_Spend': opt_spend,
            'Recommended_TCPA': opt_tcpa,
            'Expected_Conversions': opt_convs,
            'Confidence': r2
        })
    results_df = pd.DataFrame(adgroup_results)
    # Normalize spend to fit total budget
    if results_df['Recommended_Weekly_Spend'].sum() > 0:
        results_df['Recommended_Weekly_Spend'] = results_df['Recommended_Weekly_Spend'] * (weekly_budget / results_df['Recommended_Weekly_Spend'].sum())
    # Recompute expected conversions and TCPA after normalization
    results_df['Expected_Conversions'] = results_df.apply(lambda x: x['Model_a'] + x['Model_b'] * np.log(x['Recommended_Weekly_Spend']), axis=1)
    results_df['Recommended_TCPA'] = results_df['Recommended_Weekly_Spend'] / results_df['Expected_Conversions']
    # Strategic recommendations
    results_df['Recommendation'] = results_df.apply(
        lambda x: 'Increase' if x['Recommended_Weekly_Spend'] > df[df['AdGroup']==x['AdGroup']]['Spend'].mean() else 'Decrease', axis=1)
    return results_df

# --- Simulator ---
def mmm_simulator(df, custom_spend_dict):
    """
    Simulate conversions and TCPA for a custom weekly spend allocation dict: {adgroup: spend}
    """
    sim_results = []
    for adgroup, group_df in df.groupby('AdGroup'):
        model, a, b, r2 = fit_saturation_model(group_df)
        if adgroup not in custom_spend_dict or np.isnan(a) or np.isnan(b) or b <= 0:
            continue
        spend = custom_spend_dict[adgroup]
        conversions = a + b * np.log(spend)
        tcpas = spend / conversions if conversions > 0 else np.nan
        sim_results.append({
            'AdGroup': adgroup,
            'Simulated_Spend': spend,
            'Simulated_Conversions': conversions,
            'Simulated_TCPA': tcpas,
            'Confidence': r2
        })
    return pd.DataFrame(sim_results)

# --- Main Entrypoint ---
def main():
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['week_start'] = pd.to_datetime(df['week_start'])
    logger.info(f"Loaded {len(df)} rows.")
    results_df = mmm_optimize_weekly_budget(df)
    results_df.to_csv('mmm_weekly_budget_recommendations.csv', index=False)
    logger.info("Saved recommendations to mmm_weekly_budget_recommendations.csv")
    # Print summary
    print(results_df[['AdGroup','Recommended_Weekly_Spend','Recommended_TCPA','Expected_Conversions','Confidence','Recommendation']])
    # Example: run a simulation with a custom allocation
    custom_spend = {row['AdGroup']: row['Recommended_Weekly_Spend'] for _, row in results_df.iterrows()}
    sim_df = mmm_simulator(df, custom_spend)
    sim_df.to_csv('mmm_simulation_results.csv', index=False)
    logger.info("Saved simulation results to mmm_simulation_results.csv")

if __name__ == '__main__':
    main()
