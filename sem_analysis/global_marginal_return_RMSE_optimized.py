import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class GlobalMarginalReturnOptimizerRMSE:
    def __init__(self, data, target_col='Conversions', spend_col='Spend', 
                 min_spend=0.1, max_spend=1000, step=0.1, confidence_threshold=0.7):
        self.data = data
        self.target_col = target_col
        self.spend_col = spend_col
        self.min_spend = min_spend
        self.max_spend = max_spend
        self.step = step
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        self.performance_plots = {}
        
    def engineer_features(self, df):
        """Engineer additional features to improve model performance"""
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Convert timestamp to numeric features if it exists
        if 'week_start' in df.columns:
            df['month'] = df['week_start'].dt.month
            df['quarter'] = df['week_start'].dt.quarter
            df['day_of_week'] = df['week_start'].dt.dayofweek
            df['week_of_year'] = df['week_start'].dt.isocalendar().week
            # Drop the original timestamp column
            df = df.drop('week_start', axis=1)
        
        # Basic spend features
        df['spend_squared'] = df[self.spend_col] ** 2
        df['spend_cubed'] = df[self.spend_col] ** 3
        df['spend_log'] = np.log1p(df[self.spend_col])
        
        # CTR and CVR metrics
        df['CTR'] = df['Clicks'] / df['Impressions']
        df['CVR'] = df['Conversions'] / df['Clicks']
        
        # CTR and CVR trends
        df['CTR_velocity'] = df['CTR'].diff()
        df['CVR_velocity'] = df['CVR'].diff()
        
        # CTR and CVR moving averages
        for window in [3, 7, 14]:
            df[f'CTR_ma_{window}'] = df['CTR'].rolling(window=window).mean()
            df[f'CVR_ma_{window}'] = df['CVR'].rolling(window=window).mean()
        
        # Spend velocity and trends
        df['spend_velocity'] = df[self.spend_col].diff()
        df['spend_acceleration'] = df['spend_velocity'].diff()
        
        # Conversion efficiency metrics
        df['conversion_efficiency'] = df['Conversions'] / df[self.spend_col]
        df['spend_conv_ratio'] = df[self.spend_col] / df['Conversions']
        
        # Moving averages with different windows
        for window in [3, 7, 14]:
            df[f'spend_ma_{window}'] = df[self.spend_col].rolling(window=window).mean()
            df[f'conversion_ma_{window}'] = df['Conversions'].rolling(window=window).mean()
            df[f'spend_conv_ratio_ma_{window}'] = df['spend_conv_ratio'].rolling(window=window).mean()
        
        # Volatility metrics
        df['spend_volatility'] = df[self.spend_col].rolling(window=3).std()
        df['conversion_volatility'] = df['Conversions'].rolling(window=3).std()
        df['CTR_volatility'] = df['CTR'].rolling(window=3).std()
        df['CVR_volatility'] = df['CVR'].rolling(window=3).std()
        
        # Fill any NaN values with 0
        df = df.fillna(0)
        
        # Ensure all columns are numeric
        for col in df.columns:
            try:
                # Try to convert to float, which handles both numeric and nullable integer types
                df[col] = df[col].astype(float)
            except (ValueError, TypeError):
                # If conversion fails, try to convert to numeric with coerce
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        return df
    
    def calculate_enhanced_metrics(self, y_true, y_pred, X):
        """Calculate comprehensive model performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['R2'] = r2_score(y_true, y_pred)
        
        # Advanced metrics
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.mean(y_true) > 0 else float('inf')
        metrics['NRMSE'] = metrics['RMSE'] / np.mean(y_true) * 100 if np.mean(y_true) > 0 else float('inf')
        metrics['Stability'] = 1 - (np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0
        
        # Direction accuracy
        if len(y_true) > 1:
            metrics['Direction_Accuracy'] = np.mean(
                np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
            )
        else:
            metrics['Direction_Accuracy'] = 0
        
        # CTR and CVR impact metrics
        if 'CTR' in X.columns and 'CVR' in X.columns:
            metrics['CTR_Impact'] = np.corrcoef(X['CTR'], y_true)[0,1]
            metrics['CVR_Impact'] = np.corrcoef(X['CVR'], y_true)[0,1]
        
        return metrics
    
    def calculate_business_metrics(self, current_spend, recommended_spend, 
                                 current_conversions, predicted_conversions,
                                 current_ctr=None, current_cvr=None):
        """Calculate business impact metrics"""
        metrics = {}
        
        # Spend efficiency
        spend_diff = recommended_spend - current_spend
        conv_diff = predicted_conversions - current_conversions
        
        if spend_diff != 0:
            metrics['Spend_Efficiency'] = conv_diff / spend_diff
        else:
            metrics['Spend_Efficiency'] = 0
        
        # ROI projection
        if current_conversions > 0:
            metrics['ROI_Projection'] = (conv_diff / current_conversions) * 100
        else:
            metrics['ROI_Projection'] = 0
        
        # Break-even point
        if conv_diff > 0:
            metrics['Break_Even_Point'] = current_spend / conv_diff
        else:
            metrics['Break_Even_Point'] = float('inf')
        
        # CTR and CVR metrics if available
        if current_ctr is not None:
            metrics['CTR'] = current_ctr
        if current_cvr is not None:
            metrics['CVR'] = current_cvr
        
        return metrics
    
    def calculate_enhanced_confidence(self, metrics):
        """Calculate comprehensive confidence score"""
        weights = {
            'R2': 0.25,
            'NRMSE': 0.25,
            'Stability': 0.20,
            'Direction_Accuracy': 0.15,
            'ROI_Projection': 0.15
        }
        
        # Normalize NRMSE to 0-1 scale (lower is better)
        nrmse_score = 1 - min(metrics['NRMSE']/100, 1) if metrics['NRMSE'] != float('inf') else 0
        
        # Normalize ROI projection to 0-1 scale
        roi_score = min(metrics['ROI_Projection']/100, 1) if metrics['ROI_Projection'] > 0 else 0
        
        confidence = (
            weights['R2'] * metrics['R2'] +
            weights['NRMSE'] * nrmse_score +
            weights['Stability'] * metrics['Stability'] +
            weights['Direction_Accuracy'] * metrics['Direction_Accuracy'] +
            weights['ROI_Projection'] * roi_score
        )
        
        return confidence
    
    def generate_performance_plots(self, y_true, y_pred, adgroup_name):
        """Generate performance visualization plots"""
        # Actual vs Predicted plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true, name='Actual', mode='lines'))
        fig.add_trace(go.Scatter(y=y_pred, name='Predicted', mode='lines'))
        fig.update_layout(
            title=f'Actual vs Predicted - {adgroup_name}',
            xaxis_title='Time Period',
            yaxis_title='Conversions'
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=residuals, name='Residuals', mode='lines'))
        fig2.update_layout(
            title=f'Residuals - {adgroup_name}',
            xaxis_title='Time Period',
            yaxis_title='Residual Value'
        )
        
        return fig, fig2
    
    def fit_ensemble_model(self, X, y):
        """Fit an ensemble of models for better prediction accuracy"""
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models with optimized parameters
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Fit models
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Get predictions
        rf_pred = rf_model.predict(X_val)
        gb_pred = gb_model.predict(X_val)
        
        # Calculate weights based on validation performance
        rf_r2 = r2_score(y_val, rf_pred)
        gb_r2 = r2_score(y_val, gb_pred)
        total_r2 = rf_r2 + gb_r2
        
        if total_r2 > 0:
            rf_weight = rf_r2 / total_r2
            gb_weight = gb_r2 / total_r2
        else:
            rf_weight = gb_weight = 0.5
        
        return rf_model, gb_model, rf_weight, gb_weight
    
    def predict_with_ensemble(self, X, rf_model, gb_model, rf_weight, gb_weight):
        """Make predictions using the weighted ensemble"""
        rf_pred = rf_model.predict(X)
        gb_pred = gb_model.predict(X)
        return rf_weight * rf_pred + gb_weight * gb_pred
    
    def optimize_budget(self, adgroup_data):
        """Optimize budget for a single ad group with RMSE-optimized approach"""
        if len(adgroup_data) < 3:
            return None
        
        # Engineer features
        X = self.engineer_features(adgroup_data.copy())
        y = adgroup_data[self.target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit ensemble model
        rf_model, gb_model, rf_weight, gb_weight = self.fit_ensemble_model(X_scaled, y)
        
        # Generate spend range with fewer points
        current_spend = adgroup_data[self.spend_col].iloc[-1]
        current_conversions = adgroup_data[self.target_col].iloc[-1]
        current_ctr = adgroup_data['Clicks'].iloc[-1] / adgroup_data['Impressions'].iloc[-1] if 'Clicks' in adgroup_data.columns and 'Impressions' in adgroup_data.columns else None
        current_cvr = adgroup_data['Conversions'].iloc[-1] / adgroup_data['Clicks'].iloc[-1] if 'Clicks' in adgroup_data.columns else None
        
        # Create a more focused spend range around current spend
        spend_range = np.concatenate([
            np.linspace(max(self.min_spend, current_spend * 0.5), current_spend * 0.9, 10),
            np.linspace(current_spend * 0.9, current_spend * 1.1, 5),
            np.linspace(current_spend * 1.1, min(self.max_spend, current_spend * 2), 10)
        ])
        spend_range = np.unique(spend_range)
        
        # Calculate marginal returns
        best_spend = None
        best_marginal_return = -np.inf
        best_confidence = 0
        best_metrics = {}
        best_business_metrics = {}
        
        # Early stopping variables
        no_improvement_count = 0
        max_no_improvement = 5
        last_best_marginal_return = -np.inf
        
        for i, spend in enumerate(spend_range):
            # Create prediction input
            pred_input = X.iloc[-1:].copy()
            pred_input[self.spend_col] = spend
            pred_input = self.engineer_features(pred_input)
            pred_input_scaled = self.scaler.transform(pred_input)
            
            # Get ensemble prediction
            pred_conversions = self.predict_with_ensemble(pred_input_scaled, rf_model, gb_model, rf_weight, gb_weight)[0]
            
            # Calculate metrics only if we have a reasonable prediction
            if pred_conversions > 0:
                # Calculate metrics
                y_pred = self.predict_with_ensemble(X_scaled, rf_model, gb_model, rf_weight, gb_weight)
                metrics = self.calculate_enhanced_metrics(y, y_pred, X)
                
                # Calculate business metrics
                business_metrics = self.calculate_business_metrics(
                    current_spend, spend,
                    current_conversions, pred_conversions,
                    current_ctr, current_cvr
                )
                
                # Combine metrics for confidence calculation
                combined_metrics = {**metrics, **business_metrics}
                
                # Calculate confidence
                confidence = self.calculate_enhanced_confidence(combined_metrics)
                
                # Calculate marginal return
                if spend > 0:
                    marginal_return = pred_conversions / spend
                    
                    if marginal_return > best_marginal_return and confidence >= self.confidence_threshold:
                        best_marginal_return = marginal_return
                        best_spend = spend
                        best_confidence = confidence
                        best_metrics = metrics
                        best_business_metrics = business_metrics
                        
                        # Reset no improvement counter
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Early stopping check
                    if no_improvement_count >= max_no_improvement:
                        break
        
        if best_spend is None:
            return None
        
        # Generate performance plots
        y_pred = self.predict_with_ensemble(X_scaled, rf_model, gb_model, rf_weight, gb_weight)
        self.performance_plots[adgroup_data['AdGroup'].iloc[0]] = self.generate_performance_plots(y, y_pred, adgroup_data['AdGroup'].iloc[0])
        
        # Determine confidence level
        if best_confidence >= 0.8:
            confidence_level = "High"
        elif best_confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Generate business justification
        if best_spend > current_spend:
            justification = f"Recommended to increase spend by ${best_spend - current_spend:.2f} to maximize marginal returns"
        elif best_spend < current_spend:
            justification = f"Recommended to decrease spend by ${current_spend - best_spend:.2f} to optimize efficiency"
        else:
            justification = "Current spend level is optimal"
        
        result = {
            'AdGroup': adgroup_data['AdGroup'].iloc[0],
            'Current_Spend': current_spend,
            'Current_Conversions': current_conversions,
            'Current_TCPA': current_spend / current_conversions if current_conversions > 0 else float('inf'),
            'Recommended_Spend': best_spend,
            'Expected_Conversions': pred_conversions,
            'Expected_TCPA': best_spend / pred_conversions if pred_conversions > 0 else float('inf'),
            'Marginal_Conversion_per_Dollar': best_marginal_return,
            'Confidence': best_confidence,
            'Confidence_Level': confidence_level,
            'R2': best_metrics['R2'],
            'RMSE': best_metrics['RMSE'],
            'MAPE': best_metrics['MAPE'],
            'NRMSE': best_metrics['NRMSE'],
            'Stability': best_metrics['Stability'],
            'Direction_Accuracy': best_metrics['Direction_Accuracy'],
            'Spend_Efficiency': best_business_metrics['Spend_Efficiency'],
            'ROI_Projection': best_business_metrics['ROI_Projection'],
            'Break_Even_Point': best_business_metrics['Break_Even_Point'],
            'Business_Justification': justification
        }
        
        # Add CTR and CVR metrics if available
        if current_ctr is not None:
            result['CTR'] = current_ctr
        if current_cvr is not None:
            result['CVR'] = current_cvr
        if 'CTR_Impact' in best_metrics:
            result['CTR_Impact'] = best_metrics['CTR_Impact']
        if 'CVR_Impact' in best_metrics:
            result['CVR_Impact'] = best_metrics['CVR_Impact']
        
        return result
    
    def optimize_all_adgroups(self):
        """Optimize budgets for all ad groups"""
        results = []
        
        for adgroup in self.data['AdGroup'].unique():
            adgroup_data = self.data[self.data['AdGroup'] == adgroup].copy()
            result = self.optimize_budget(adgroup_data)
            if result:
                results.append(result)
        
        return pd.DataFrame(results)
