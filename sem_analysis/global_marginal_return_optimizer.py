import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEEKLY_BUDGET = 40000
DATA_PATH = 'sem_analysis/SEM_DATA_TOP100.csv'

# --- Helper Functions ---
def fit_log_model(group_df):
    df = group_df[group_df['Spend'] > 0].copy()
    if len(df) < 2:
        return None, np.nan, np.nan, np.nan
    X = np.log(df['Spend']).values.reshape(-1, 1)
    X = sm.add_constant(X)
    y = df['Conversions'].values
    model = sm.OLS(y, X).fit()
    a, b = model.params
    return model, a, b, model.rsquared

def global_marginal_return_optimizer(df, total_budget=WEEKLY_BUDGET):
    logger.info(f"Global marginal return optimization for total budget ${total_budget:,.0f}")
    adgroup_params = []
    adgroup_indices = []  # For explicit indexing
    for idx, (adgroup, group_df) in enumerate(df.groupby('AdGroup')):
        model, a, b, r2 = fit_log_model(group_df)
        if np.isnan(a) or np.isnan(b) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            continue
        # Current metrics
        current_spend = group_df['Spend'].mean()
        logger.info(f"AdGroup: {adgroup}, Index: {idx}, Current Spend: {current_spend}")  # Debug log
        current_conversions = group_df['Conversions'].mean()
        current_tcpa = current_spend / current_conversions if current_conversions > 0 else np.nan
        impressions = group_df['Impressions'].mean()
        clicks = group_df['Clicks'].mean()
        ctr = clicks / impressions if impressions > 0 else 0
        cvr = current_conversions / clicks if clicks > 0 else 0
        adgroup_params.append({
            'AdGroup_Index': idx,
            'AdGroup': adgroup,
            'a': a,
            'b': b,
            'R2': r2,
            'Current_Spend': current_spend,
            'Current_Conversions': current_conversions,
            'Current_TCPA': current_tcpa,
            'Impressions': impressions,
            'Clicks': clicks,
            'CTR': ctr,
            'CVR': cvr
        })
        adgroup_indices.append(idx)
    params_df = pd.DataFrame(adgroup_params)
    n = len(params_df)
    if n == 0:
        logger.error("No valid adgroups for optimization.")
        return pd.DataFrame()
    # Initial guess: proportional to b
    x0 = np.ones(n) * (total_budget / n)
    bounds = [(1, total_budget) for _ in range(n)]
    def objective(spends):
        # Negative total conversions (since we minimize)
        return -np.sum(params_df['a'].values + params_df['b'].values * np.log(spends))
    def constraint(spends):
        return np.sum(spends) - total_budget
    constraints = {'type': 'eq', 'fun': constraint}
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    if not result.success:
        logger.error(f"Optimization failed: {result.message}")
        return pd.DataFrame()
    spends = result.x
    conversions = params_df['a'].values + params_df['b'].values * np.log(spends)
    marginal_returns = params_df['b'].values / spends
    expected_tcpa = spends / conversions
    out_df = params_df.copy()
    out_df['Recommended_Spend'] = spends
    out_df['Expected_Conversions'] = conversions
    out_df['Expected_TCPA'] = expected_tcpa
    out_df['Marginal_Conversion_per_Dollar'] = marginal_returns
    out_df['Confidence'] = params_df['R2']
    # Business justification
    justifications = []
    for i, row in out_df.iterrows():
        if row['Recommended_Spend'] > row['Current_Spend']:
            reason = (
                f"Increase spend because this AdGroup has {row['Impressions']:.0f} avg weekly impressions, "
                f"{row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), and {row['Current_Conversions']:.2f} conversions (CVR: {row['CVR']:.2%}). "
                f"Current TCPA is ${row['Current_TCPA']:.2f}, expected TCPA is ${row['Expected_TCPA']:.2f}. "
                f"The marginal conversion per dollar is {row['Marginal_Conversion_per_Dollar']:.4f}, indicating room for profitable scaling. "
                f"The model does not show strong saturation at current spend, so additional budget is likely to yield incremental conversions."
            )
        else:
            reason = (
                f"Decrease spend because this AdGroup, despite {row['Impressions']:.0f} avg weekly impressions and {row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), "
                f"shows signs of saturation or inefficiency. Current TCPA is ${row['Current_TCPA']:.2f}, expected TCPA is ${row['Expected_TCPA']:.2f}. "
                f"The marginal conversion per dollar is {row['Marginal_Conversion_per_Dollar']:.4f}, which is relatively low, indicating diminishing returns. "
                f"The model suggests that reallocating budget elsewhere will yield better overall results."
            )
        justifications.append(reason)
    out_df['Business_Justification'] = justifications
    # Reorder columns for clarity, include AdGroup_Index
    out_df = out_df[['AdGroup_Index','AdGroup','Current_Spend','Current_Conversions','Current_TCPA',
                     'Recommended_Spend','Expected_Conversions','Expected_TCPA',
                     'Marginal_Conversion_per_Dollar','Confidence','Business_Justification']]
    return out_df

def plot_response_curves_conversions(df, results_df, output_dir='response_curves_conversions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for _, row in results_df.iterrows():
        adgroup = row['AdGroup']
        group_df = df[df['AdGroup'] == adgroup].copy()
        group_df = group_df[group_df['Spend'] > 0]
        if len(group_df) < 2:
            continue
        X = np.log(group_df['Spend']).values.reshape(-1, 1)
        X = sm.add_constant(X)
        y = group_df['Conversions'].values
        model = sm.OLS(y, X).fit()
        a, b = model.params
        spend_range = np.linspace(group_df['Spend'].min(), group_df['Spend'].max() * 1.5, 100)
        conv_pred = a + b * np.log(spend_range)
        plt.figure(figsize=(10, 7))
        plt.scatter(group_df['Spend'], group_df['Conversions'], color='blue', alpha=0.5, label='Historical Data')
        plt.plot(spend_range, conv_pred, 'r-', label='Fitted Curve')
        plt.scatter(row['Current_Spend'], row['Current_Conversions'], color='orange', s=100, label='Current Avg')
        plt.scatter(row['Recommended_Spend'], row['Expected_Conversions'], color='green', s=100, label='Recommended')
        plt.xlabel('Weekly Spend')
        plt.ylabel('Weekly Conversions')
        plt.title(f'Conversion Response Curve - {adgroup}')
        plt.legend()
        plt.grid(True)
        # Add business justification as annotation
        plt.gcf().text(0.02, 0.02, row['Business_Justification'], fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{adgroup}_conversion_response_curve.png')
        plt.close()

def main():
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['week_start'] = pd.to_datetime(df['week_start'])
    logger.info(f"Loaded {len(df)} rows.")
    results_df = global_marginal_return_optimizer(df, WEEKLY_BUDGET)
    if not results_df.empty:
        results_df.to_csv('global_marginal_return_optimization.csv', index=False)
        logger.info("Saved results to global_marginal_return_optimization.csv")
        print(results_df)
        plot_response_curves_conversions(df, results_df)
        logger.info("Saved response curve plots to response_curves_conversions/")
    else:
        print("No valid optimization results.")

if __name__ == '__main__':
    main()
