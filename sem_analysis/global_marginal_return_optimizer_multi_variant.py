import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

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
    adgroup_rmses = []
    skipped_adgroups = []
    for idx, (adgroup, group_df) in enumerate(df.groupby('AdGroup')):
        logger.info(f"\nProcessing AdGroup {idx}: {adgroup}")
        logger.info(f"Group data shape: {group_df.shape}")
        model, a, b, c, d, r2 = fit_multivariate_model(group_df)
        if np.isnan(a) or np.isnan(b) or np.isnan(c) or np.isnan(d) or b <= 0:
            logger.warning(f"Skipping {adgroup} due to insufficient or non-positive model fit.")
            skipped_adgroups.append({
                'AdGroup': adgroup,
                'Reason': 'Insufficient or non-positive model fit (need more data or better data quality)'
            })
            continue
        # Calculate per-adgroup RMSE
        X = pd.DataFrame({
            'log_Spend': np.log(group_df['Spend']),
            'CTR': group_df['Clicks'] / group_df['Impressions'],
            'CVR': group_df['Conversions'] / group_df['Clicks']
        })
        X = sm.add_constant(X)
        y_true = group_df['Conversions'].values
        y_pred = a + b * X['log_Spend'] + c * X['CTR'] + d * X['CVR']
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(mask) > 0:
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        else:
            rmse = np.nan
        adgroup_rmses.append({'AdGroup': adgroup, 'RMSE': rmse, 'n': len(y_true)})
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
            'CVR': cvr,
            'RMSE': rmse,
        })
        adgroup_indices.append(idx)
    params_df = pd.DataFrame(adgroup_params)
    n = len(params_df)
    if n == 0:
        logger.error("No valid adgroups for optimization.")
        return pd.DataFrame(), skipped_adgroups, adgroup_rmses
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
            return pd.DataFrame(), skipped_adgroups, adgroup_rmses
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
        # Add business justification
        justifications = []
        for i, row in out_df.iterrows():
            if row['Recommended_Spend'] > row['Current_Spend']:
                reason = (
                    f"Increase spend because this AdGroup has {row['Impressions']:.0f} avg impressions, "
                    f"{row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), and {row['Current_Conversions']:.2f} conversions (CVR: {row['CVR']:.2%}). "
                    f"Current TCPA is ${row['Current_TCPA']:.2f}, expected TCPA is ${row['Expected_TCPA']:.2f}. "
                    f"The marginal conversion per dollar is {row['Marginal_Conversion_per_Dollar']:.4f}. "
                    f"The multivariate model (using Spend, CTR, CVR) suggests that additional budget is likely to yield incremental conversions. "
                    f"Model coefficients: log(Spend)={row['b']:.3f}, CTR={row['c']:.3f}, CVR={row['d']:.3f}. "
                    f"Confidence: {row['Confidence_Level']} (R²={row['Confidence']:.2%})."
                )
            else:
                reason = (
                    f"Decrease spend because this AdGroup, despite {row['Impressions']:.0f} avg impressions and {row['Clicks']:.0f} clicks (CTR: {row['CTR']:.2%}), "
                    f"shows signs of saturation or inefficiency. Current TCPA is ${row['Current_TCPA']:.2f}, expected TCPA is ${row['Expected_TCPA']:.2f}. "
                    f"The marginal conversion per dollar is {row['Marginal_Conversion_per_Dollar']:.4f}. "
                    f"The multivariate model (using Spend, CTR, CVR) suggests reallocating budget elsewhere will yield better overall results. "
                    f"Model coefficients: log(Spend)={row['b']:.3f}, CTR={row['c']:.3f}, CVR={row['d']:.3f}. "
                    f"Confidence: {row['Confidence_Level']} (R²={row['Confidence']:.2%})."
                )
            justifications.append(reason)
        out_df['Business_Justification'] = justifications
        out_df['RMSE'] = params_df['RMSE']
        out_df = out_df[['AdGroup_Index','AdGroup','Current_Spend','Current_Conversions','Current_TCPA',
                         'Recommended_Spend','Expected_Conversions','Expected_TCPA',
                         'Marginal_Conversion_per_Dollar','Confidence','Confidence_Level','CTR','CVR','Business_Justification','RMSE']]
        logger.info("Optimization complete. Returning results.")
        return out_df, skipped_adgroups, adgroup_rmses
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return pd.DataFrame(), skipped_adgroups, adgroup_rmses
