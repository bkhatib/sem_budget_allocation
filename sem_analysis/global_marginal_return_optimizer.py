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
        logger.warning(f"Insufficient data points for modeling: {len(df)} points")
        return None, np.nan, np.nan, np.nan
    
    X = np.log(df['Spend']).values.reshape(-1, 1)
    X = sm.add_constant(X)
    y = df['Conversions'].values
    
    try:
        model = sm.OLS(y, X).fit()
        a, b = model.params
        logger.info(f"Model fit successful - a: {a:.4f}, b: {b:.4f}, R2: {model.rsquared:.4f}")
        return model, a, b, model.rsquared
    except Exception as e:
        logger.error(f"Error fitting model: {str(e)}")
        return None, np.nan, np.nan, np.nan

def global_marginal_return_optimizer(df, total_budget=WEEKLY_BUDGET):
    logger.info(f"Starting global marginal return optimization for total budget ${total_budget:,.0f}")
    logger.info(f"Input data shape: {df.shape}")
    
    adgroup_params = []
    adgroup_indices = []  # For explicit indexing
    
    # Group by AdGroup and process each group
    for idx, (adgroup, group_df) in enumerate(df.groupby('AdGroup')):
        logger.info(f"\nProcessing AdGroup {idx}: {adgroup}")
        logger.info(f"Group data shape: {group_df.shape}")
        
        model, a, b, r2 = fit_log_model(group_df)
        
        if np.isnan(a) or np.isnan(b) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            continue
            
        # Current metrics
        current_spend = group_df['Spend'].mean()
        current_conversions = group_df['Conversions'].mean()
        current_tcpa = current_spend / current_conversions if current_conversions > 0 else np.nan
        impressions = group_df['Impressions'].mean()
        clicks = group_df['Clicks'].mean()
        ctr = clicks / impressions if impressions > 0 else 0
        cvr = current_conversions / clicks if clicks > 0 else 0
        
        logger.info(f"Current metrics for {adgroup}:")
        logger.info(f"  Spend: ${current_spend:.2f}")
        logger.info(f"  Conversions: {current_conversions:.2f}")
        logger.info(f"  TCPA: ${current_tcpa:.2f}")
        logger.info(f"  CTR: {ctr:.2%}")
        logger.info(f"  CVR: {cvr:.2%}")
        
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
        
    logger.info(f"\nOptimization parameters:")
    logger.info(f"Number of valid adgroups: {n}")
    logger.info(f"Total budget: ${total_budget:,.2f}")
    
    # Initial guess: proportional to current spend
    current_spends = params_df['Current_Spend'].values
    total_current_spend = np.sum(current_spends)
    if total_current_spend > 0:
        x0 = current_spends * (total_budget / total_current_spend)
    else:
        x0 = np.ones(n) * (total_budget / n)
    
    # Set bounds with a minimum spend of $1
    bounds = [(1, total_budget) for _ in range(n)]
    
    def objective(spends):
        # Negative total conversions (since we minimize)
        return -np.sum(params_df['a'].values + params_df['b'].values * np.log(spends))
    
    def constraint(spends):
        return np.sum(spends) - total_budget
    
    constraints = {'type': 'eq', 'fun': constraint}
    
    try:
        # Optimization options
        options = {
            'maxiter': 1000,  # Increase maximum iterations
            'ftol': 1e-8,     # Tighter function tolerance
            'disp': True      # Show optimization progress
        }
        
        result = minimize(
            objective, 
            x0, 
            bounds=bounds, 
            constraints=constraints, 
            method='SLSQP',
            options=options
        )
        
        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            # If optimization fails, try with a different initial guess
            logger.info("Trying optimization with alternative initial guess...")
            x0_alt = np.ones(n) * (total_budget / n)
            result = minimize(
                objective, 
                x0_alt, 
                bounds=bounds, 
                constraints=constraints, 
                method='SLSQP',
                options=options
            )
            if not result.success:
                logger.error(f"Optimization failed again: {result.message}")
                return pd.DataFrame()
            
        logger.info("Optimization successful!")
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
        
        # Reorder columns for clarity
        out_df = out_df[['AdGroup_Index','AdGroup','Current_Spend','Current_Conversions','Current_TCPA',
                         'Recommended_Spend','Expected_Conversions','Expected_TCPA',
                         'Marginal_Conversion_per_Dollar','Confidence','Business_Justification']]
        
        logger.info("Optimization complete. Returning results.")
        return out_df
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return pd.DataFrame()

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
