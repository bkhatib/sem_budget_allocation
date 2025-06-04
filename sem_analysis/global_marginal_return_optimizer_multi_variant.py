import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEEKLY_BUDGET = 40000
DATA_PATH = 'sem_analysis/SEM_DATA_TOP100.csv'

def fit_multivariate_model(group_df):
    df = group_df[(group_df['Spend'] > 0) & (group_df['Clicks'] > 0)].copy()
    if len(df) < 2:
        logger.warning(f"Insufficient data points for modeling: {len(df)} points")
        return None, np.nan, np.nan, np.nan, np.nan, np.nan
    X = pd.DataFrame({
        'log_Spend': np.log(df['Spend']),
        'CTR': df['Clicks'] / df['Impressions'],
        'CVR': df['Conversions'] / df['Clicks']
    })
    X = sm.add_constant(X)
    y = df['Conversions'].values
    try:
        model = sm.OLS(y, X).fit()
        a = model.params['const']
        b = model.params['log_Spend']
        c = model.params['CTR']
        d = model.params['CVR']
        r2 = model.rsquared
        logger.info(f"Model fit successful - a: {a:.4f}, b: {b:.4f}, c: {c:.4f}, d: {d:.4f}, R2: {r2:.4f}")
        return model, a, b, c, d, r2
    except Exception as e:
        logger.error(f"Error fitting model: {str(e)}")
        return None, np.nan, np.nan, np.nan, np.nan, np.nan

def global_marginal_return_optimizer_multi_variant(df, total_budget=WEEKLY_BUDGET):
    logger.info(f"Starting multivariate marginal return optimization for total budget ${total_budget:,.0f}")
    logger.info(f"Input data shape: {df.shape}")
    adgroup_params = []
    adgroup_indices = []
    for idx, (adgroup, group_df) in enumerate(df.groupby('AdGroup')):
        logger.info(f"\nProcessing AdGroup {idx}: {adgroup}")
        logger.info(f"Group data shape: {group_df.shape}")
        model, a, b, c, d, r2 = fit_multivariate_model(group_df)
        if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            continue
        current_spend = group_df['Spend'].mean()
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
            'c': c,
            'd': d,
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
    # Initial guess: proportional to current spend
    current_spends = params_df['Current_Spend'].values
    total_current_spend = np.sum(current_spends)
    if total_current_spend > 0:
        x0 = current_spends * (total_budget / total_current_spend)
    else:
        x0 = np.ones(n) * (total_budget / n)
    bounds = [(1, total_budget) for _ in range(n)]
    # Use average CTR and CVR for each ad group (fixed for optimization)
    avg_ctrs = params_df['CTR'].values
    avg_cvrs = params_df['CVR'].values
    def objective(spends):
        # Negative total conversions (since we minimize)
        return -np.sum(
            params_df['a'].values +
            params_df['b'].values * np.log(spends) +
            params_df['c'].values * avg_ctrs +
            params_df['d'].values * avg_cvrs
        )
    def constraint(spends):
        return np.sum(spends) - total_budget
    constraints = {'type': 'eq', 'fun': constraint}
    try:
        options = {'maxiter': 1000, 'ftol': 1e-8, 'disp': True}
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
            return pd.DataFrame()
        spends = result.x
        conversions = (
            params_df['a'].values +
            params_df['b'].values * np.log(spends) +
            params_df['c'].values * avg_ctrs +
            params_df['d'].values * avg_cvrs
        )
        marginal_returns = params_df['b'].values / spends
        expected_tcpa = spends / conversions
        out_df = params_df.copy()
        out_df['Recommended_Spend'] = spends
        out_df['Expected_Conversions'] = conversions
        out_df['Expected_TCPA'] = expected_tcpa
        out_df['Marginal_Conversion_per_Dollar'] = marginal_returns
        out_df['Confidence'] = params_df['R2']
        out_df['Confidence_Level'] = out_df['Confidence'].apply(
            lambda r2: "Low Confidence" if r2 < 0.3 else ("Moderate Confidence" if r2 < 0.6 else "High Confidence")
        )
        out_df = out_df[['AdGroup_Index','AdGroup','Current_Spend','Current_Conversions','Current_TCPA',
                         'Recommended_Spend','Expected_Conversions','Expected_TCPA',
                         'Marginal_Conversion_per_Dollar','Confidence','Confidence_Level','CTR','CVR']]
        logger.info("Optimization complete. Returning results.")
        return out_df
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return pd.DataFrame()
