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
MAX_TCPA_FACTOR = 5.0  # Allow at most 5x the current TCPA per AdGroup (increased for quick test)
FALLBACK_MAX_TCPA = 50000  # Fallback if current TCPA is NaN or zero (increased for quick test)

# --- Helper Functions ---
def fit_log_model_revenue(group_df):
    df = group_df[group_df['Spend'] > 0].copy()
    if len(df) < 2:
        return None, np.nan, np.nan, np.nan
    X = np.log(df['Spend']).values.reshape(-1, 1)
    X = sm.add_constant(X)
    y = df['Revenue'].values
    model = sm.OLS(y, X).fit()
    a, b = model.params
    return model, a, b, model.rsquared

def global_marginal_return_optimizer_revenue(df, total_budget=WEEKLY_BUDGET):
    logger.info(f"Global marginal return optimization for total budget ${total_budget:,.0f} (Revenue)")
    adgroup_params = []
    skipped_adgroups = []
    infeasible_adgroups = []
    for adgroup, group_df in df.groupby('AdGroup'):
        model, a, b, r2 = fit_log_model_revenue(group_df)
        if np.isnan(a) or np.isnan(b) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            skipped_adgroups.append(adgroup)
            continue
        # Current metrics
        current_spend = group_df['Spend'].mean()
        current_revenue = group_df['Revenue'].mean()
        current_conversions = group_df['Conversions'].mean()
        current_roas = current_revenue / current_spend if current_spend > 0 else np.nan
        current_tcpa = current_spend / current_conversions if current_conversions > 0 else np.nan
        impressions = group_df['Impressions'].mean()
        clicks = group_df['Clicks'].mean()
        conversions = current_conversions
        ctr = clicks / impressions if impressions > 0 else 0
        cvr = conversions / clicks if clicks > 0 else 0
        # Set max TCPA for this adgroup
        max_tcpa = MAX_TCPA_FACTOR * current_tcpa if current_tcpa and not np.isnan(current_tcpa) and current_tcpa > 0 else FALLBACK_MAX_TCPA
        # Calculate min possible TCPA at min spend
        min_spend = 1.0
        # Estimate expected conversions at min spend using historical CVR
        min_expected_conversions = cvr * (min_spend / current_spend) * conversions if current_spend > 0 else 0
        min_possible_tcpa = min_spend / min_expected_conversions if min_expected_conversions > 0 else np.inf
        if min_possible_tcpa > max_tcpa:
            logger.warning(f"Excluding {adgroup} from optimization: min possible TCPA ({min_possible_tcpa:.2f}) > max allowed TCPA ({max_tcpa:.2f})")
            infeasible_adgroups.append(adgroup)
            continue
        adgroup_params.append({
            'AdGroup': adgroup,
            'a': a,
            'b': b,
            'R2': r2,
            'Current_Spend': current_spend,
            'Current_Revenue': current_revenue,
            'Current_ROAS': current_roas,
            'Current_Conversions': current_conversions,
            'Current_TCPA': current_tcpa,
            'Impressions': impressions,
            'Clicks': clicks,
            'Conversions': conversions,
            'CTR': ctr,
            'CVR': cvr,
            'Max_TCPA': max_tcpa
        })
    params_df = pd.DataFrame(adgroup_params)
    n = len(params_df)
    logger.info(f"AdGroups included in optimization: {list(params_df['AdGroup'])}")
    logger.info(f"Number of AdGroups included: {n}, skipped: {len(skipped_adgroups)}, infeasible: {len(infeasible_adgroups)}")
    if n == 0:
        logger.error("No valid adgroups for optimization.")
        return pd.DataFrame()
    # Initial guess: proportional to b
    x0 = np.ones(n) * (total_budget / n)
    bounds = [(1, total_budget) for _ in range(n)]
    logger.info(f"Initial guess for spends: {x0}")
    logger.info(f"Bounds: {bounds}")
    def objective(spends):
        return -np.sum(params_df['a'].values + params_df['b'].values * np.log(spends))
    def constraint_budget(spends):
        return np.sum(spends) - total_budget
    def constraint_tcpa(spends):
        expected_conversions = params_df['CVR'].values * (spends / (params_df['Current_Spend'].replace(0, np.nan))) * params_df['Current_Conversions'].values
        expected_tcpa = spends / expected_conversions
        return params_df['Max_TCPA'].values - expected_tcpa
    constraints = [
        {'type': 'eq', 'fun': constraint_budget},
        {'type': 'ineq', 'fun': constraint_tcpa}
    ]
    result = minimize(
        objective, x0, bounds=bounds, constraints=constraints, method='SLSQP',
        options={'maxiter': 5000, 'ftol': 1e-7, 'disp': True}
    )
    if not result.success:
        logger.error(f"Optimization failed: {result.message}")
        logger.error(f"Last tried spends: {result.x if hasattr(result, 'x') else 'N/A'}")
        logger.error(f"Objective value at last step: {objective(result.x) if hasattr(result, 'x') else 'N/A'}")
        return pd.DataFrame()
    spends = result.x
    revenues = params_df['a'].values + params_df['b'].values * np.log(spends)
    marginal_returns = params_df['b'].values / spends
    expected_roas = revenues / spends
    expected_conversions = params_df['CVR'].values * (spends / (params_df['Current_Spend'].replace(0, np.nan))) * params_df['Current_Conversions'].values
    expected_tcpa = spends / expected_conversions
    # Log any adgroups that hit the constraint
    for i, (tcpa, max_tcpa, adg) in enumerate(zip(expected_tcpa, params_df['Max_TCPA'].values, params_df['AdGroup'])):
        if np.isclose(tcpa, max_tcpa, atol=1e-2) or tcpa > max_tcpa:
            logger.info(f"AdGroup {adg} hits TCPA constraint: expected TCPA={tcpa:.2f}, max TCPA={max_tcpa:.2f}")
    # Business justification
    justifications = []
    for i, row in params_df.iterrows():
        rec_spend = spends[i]
        exp_rev = revenues[i]
        exp_roas = expected_roas[i]
        marg = marginal_returns[i]
        curr_tcpa = row['Current_TCPA']
        exp_tcpa = expected_tcpa[i]
        if rec_spend > row['Current_Spend']:
            reason = (
                f"Increase spend because this AdGroup has {row['Impressions']:.0f} avg weekly impressions, "
                f"{row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), {row['Conversions']:.2f} conversions (CVR: {row['CVR']:.2%}), and current ROAS ${row['Current_ROAS']:.2f}, TCPA ${curr_tcpa:.2f}. "
                f"Expected ROAS is ${exp_roas:.2f}, expected TCPA is ${exp_tcpa:.2f}. Marginal revenue per dollar is {marg:.4f}, indicating room for profitable scaling. "
                f"The model does not show strong saturation at current spend, so additional budget is likely to yield incremental revenue."
            )
        else:
            reason = (
                f"Decrease spend because this AdGroup, despite {row['Impressions']:.0f} avg weekly impressions and {row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), "
                f"shows signs of saturation or inefficiency. Current ROAS is ${row['Current_ROAS']:.2f}, TCPA ${curr_tcpa:.2f}, expected ROAS is ${exp_roas:.2f}, expected TCPA is ${exp_tcpa:.2f}. "
                f"Marginal revenue per dollar is {marg:.4f}, which is relatively low, indicating diminishing returns. "
                f"The model suggests that reallocating budget elsewhere will yield better overall results."
            )
        justifications.append(reason)
    out_df = params_df.copy()
    out_df['Recommended_Spend'] = spends
    out_df['Expected_Revenue'] = revenues
    out_df['Expected_ROAS'] = expected_roas
    out_df['Expected_TCPA'] = expected_tcpa
    out_df['Marginal_Revenue_per_Dollar'] = marginal_returns
    out_df['Business_Justification'] = justifications
    out_df['Spend_Change'] = out_df['Recommended_Spend'] - out_df['Current_Spend']
    # Reorder columns for clarity
    out_df = out_df[['AdGroup','Current_Spend','Current_Revenue','Current_ROAS','Current_Conversions','Current_TCPA',
                     'Recommended_Spend','Expected_Revenue','Expected_ROAS','Expected_TCPA',
                     'Spend_Change','Marginal_Revenue_per_Dollar','R2','Business_Justification']]
    return out_df

def plot_response_curves_revenue(df, results_df, output_dir='response_curves_revenue'):
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
        y = group_df['Revenue'].values
        model = sm.OLS(y, X).fit()
        a, b = model.params
        spend_range = np.linspace(group_df['Spend'].min(), group_df['Spend'].max() * 1.5, 100)
        revenue_pred = a + b * np.log(spend_range)
        plt.figure(figsize=(10, 7))
        plt.scatter(group_df['Spend'], group_df['Revenue'], color='blue', alpha=0.5, label='Historical Data')
        plt.plot(spend_range, revenue_pred, 'r-', label='Fitted Curve')
        plt.scatter(row['Current_Spend'], row['Current_Revenue'], color='orange', s=100, label='Current Avg')
        plt.scatter(row['Recommended_Spend'], row['Expected_Revenue'], color='green', s=100, label='Recommended')
        plt.xlabel('Weekly Spend')
        plt.ylabel('Weekly Revenue')
        plt.title(f'Revenue Response Curve - {adgroup}')
        plt.legend()
        plt.grid(True)
        # Add business justification as annotation
        plt.gcf().text(0.02, 0.02, row['Business_Justification'], fontsize=9, wrap=True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{adgroup}_revenue_response_curve.png')
        plt.close()

def main():
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['week_start'] = pd.to_datetime(df['week_start'])
    logger.info(f"Loaded {len(df)} rows.")
    results_df = global_marginal_return_optimizer_revenue(df, WEEKLY_BUDGET)
    if not results_df.empty:
        results_df.to_csv('global_marginal_return_optimization_revenue.csv', index=False)
        logger.info("Saved results to global_marginal_return_optimization_revenue.csv")
        print(results_df)
        plot_response_curves_revenue(df, results_df)
        logger.info("Saved response curve plots to response_curves_revenue/")
    else:
        print("No valid optimization results.")

if __name__ == '__main__':
    main()
