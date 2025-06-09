import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GlobalMarginalReturnOptimizerRMSE:
    def __init__(self, data, target_col='Conversions', spend_col='Spend', 
                 min_spend=0.1, max_spend=100, step=1.0, confidence_threshold=0.5):
        self.data = data
        self.target_col = target_col
        self.spend_col = spend_col
        self.min_spend = min_spend
        self.max_spend = max_spend
        self.step = step  # Increased default step to reduce iterations
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        self.performance_plots = {}
        self.models = {}
        self.feature_columns = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RMSE-optimized model with {len(data)} rows of data")
        
        # Add safety checks for spend range
        if self.step <= 0:
            raise ValueError("Step size must be positive")
        if self.min_spend >= self.max_spend:
            raise ValueError("min_spend must be less than max_spend")
        
        # Calculate and log the number of iterations
        num_iterations = int((self.max_spend - self.min_spend) / self.step) + 1
        if num_iterations > 100:
            self.logger.warning(f"Large number of iterations ({num_iterations}) may impact performance. Consider increasing step size.")
        
        # Add maximum iterations safety
        self.max_iterations = min(num_iterations, 100)  # Cap at 100 iterations
        
    def engineer_features(self, df):
        """Engineer additional features to improve model performance"""
        logger.info("Starting feature engineering")
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Convert timestamp to numeric features if it exists
        if 'week_start' in df.columns:
            logger.info("Converting timestamp features")
            df['month'] = df['week_start'].dt.month
            df['quarter'] = df['week_start'].dt.quarter
            df['day_of_week'] = df['week_start'].dt.dayofweek
            df['week_of_year'] = df['week_start'].dt.isocalendar().week
            # Drop the original timestamp column
            df = df.drop('week_start', axis=1)
        
        # Basic spend features
        logger.info("Creating basic spend features")
        df['spend_squared'] = df[self.spend_col] ** 2
        df['spend_cubed'] = df[self.spend_col] ** 3
        df['spend_log'] = np.log1p(df[self.spend_col])
        
        # CTR and CVR metrics with safe division
        logger.info("Calculating CTR and CVR metrics")
        df['CTR'] = df['Clicks'].div(df['Impressions'].replace(0, np.nan)).fillna(0)
        df['CVR'] = df['Conversions'].div(df['Clicks'].replace(0, np.nan)).fillna(0)
        
        # CTR and CVR trends
        logger.info("Calculating trends and moving averages")
        df['CTR_velocity'] = df['CTR'].diff().fillna(0)
        df['CVR_velocity'] = df['CVR'].diff().fillna(0)
        
        # CTR and CVR moving averages
        for window in [3, 7, 14]:
            df[f'CTR_ma_{window}'] = df['CTR'].rolling(window=window, min_periods=1).mean().fillna(0)
            df[f'CVR_ma_{window}'] = df['CVR'].rolling(window=window, min_periods=1).mean().fillna(0)
        
        # Spend velocity and trends
        df['spend_velocity'] = df[self.spend_col].diff().fillna(0)
        df['spend_acceleration'] = df['spend_velocity'].diff().fillna(0)
        
        # Conversion efficiency metrics with safe division
        logger.info("Calculating efficiency metrics")
        df['conversion_efficiency'] = df['Conversions'].div(df[self.spend_col].replace(0, np.nan)).fillna(0)
        df['spend_conv_ratio'] = df[self.spend_col].div(df['Conversions'].replace(0, np.nan)).fillna(0)
        
        # Moving averages with different windows
        for window in [3, 7, 14]:
            df[f'spend_ma_{window}'] = df[self.spend_col].rolling(window=window, min_periods=1).mean().fillna(0)
            df[f'conversion_ma_{window}'] = df['Conversions'].rolling(window=window, min_periods=1).mean().fillna(0)
            df[f'spend_conv_ratio_ma_{window}'] = df['spend_conv_ratio'].rolling(window=window, min_periods=1).mean().fillna(0)
        
        # Volatility metrics
        logger.info("Calculating volatility metrics")
        df['spend_volatility'] = df[self.spend_col].rolling(window=3, min_periods=1).std().fillna(0)
        df['conversion_volatility'] = df['Conversions'].rolling(window=3, min_periods=1).std().fillna(0)
        df['CTR_volatility'] = df['CTR'].rolling(window=3, min_periods=1).std().fillna(0)
        df['CVR_volatility'] = df['CVR'].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Fill any NaN values with 0
        df = df.fillna(0)
        
        # Ensure all columns are numeric and within float64 limits
        logger.info("Ensuring numeric values and handling infinities")
        for col in df.columns:
            try:
                # Convert to float and handle infinities
                df[col] = df[col].astype(float)
                # Replace infinities with large but finite values
                df[col] = df[col].replace([np.inf, -np.inf], [1e9, -1e9])
                # Clip values to float64 limits
                df[col] = df[col].clip(-1e9, 1e9)
            except (ValueError, TypeError):
                # If conversion fails, try to convert to numeric with coerce
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
                # Replace infinities and clip values
                df[col] = df[col].replace([np.inf, -np.inf], [1e9, -1e9])
                df[col] = df[col].clip(-1e9, 1e9)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def calculate_enhanced_metrics(self, y_true, y_pred, X):
        """Calculate comprehensive model performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate relative RMSE (RMSE/mean)
        mean_conversions = np.mean(y_true)
        if mean_conversions > 0:
            metrics['Relative_RMSE'] = (metrics['RMSE'] / mean_conversions) * 100  # As percentage
        else:
            metrics['Relative_RMSE'] = float('inf')
            
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
        """Calculate comprehensive confidence score with more lenient weights"""
        weights = {
            'R2': 0.20,
            'Relative_RMSE': 0.20,  # Changed from NRMSE to Relative_RMSE
            'Stability': 0.20,
            'Direction_Accuracy': 0.20,
            'ROI_Projection': 0.20
        }
        
        # More lenient normalization based on relative RMSE
        # New interpretation:
        # < 10%: Excellent
        # 10-20%: Good
        # 20-30%: Moderate
        # 30-40%: Low
        # > 40%: Poor
        relative_rmse_score = 1 - min(metrics['Relative_RMSE']/40, 1) if metrics['Relative_RMSE'] != float('inf') else 0
        roi_score = min(metrics['ROI_Projection']/50, 1) if metrics['ROI_Projection'] > 0 else 0
        
        confidence = (
            weights['R2'] * max(metrics['R2'], 0) +
            weights['Relative_RMSE'] * relative_rmse_score +
            weights['Stability'] * max(metrics['Stability'], 0) +
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
        
        # Store feature columns for later use
        self.feature_columns = [col for col in X.columns if col != self.target_col]
        self.logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
        
        # Scale features while maintaining DataFrame structure
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train[self.feature_columns]),
            columns=self.feature_columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val[self.feature_columns]),
            columns=self.feature_columns,
            index=X_val.index
        )
        
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
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Get predictions
        rf_pred = rf_model.predict(X_val_scaled)
        gb_pred = gb_model.predict(X_val_scaled)
        
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
        # Ensure X is a DataFrame with the correct columns
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_columns)
        
        rf_pred = rf_model.predict(X)
        gb_pred = gb_model.predict(X)
        return rf_weight * rf_pred + gb_weight * gb_pred
    
    def _calculate_adgroup_rmse(self, adgroup_data):
        """Calculate RMSE for an ad group based on its historical performance"""
        try:
            # Calculate historical RMSE
            historical_rmse = np.sqrt(np.mean((adgroup_data['Conversions'] - adgroup_data['Conversions'].mean())**2))
            
            # Calculate relative RMSE (as percentage of mean)
            mean_conversions = adgroup_data['Conversions'].mean()
            if mean_conversions > 0:
                relative_rmse = (historical_rmse / mean_conversions) * 100  # As percentage
            else:
                relative_rmse = historical_rmse
                
            # Much more lenient RMSE thresholds based on data volume
            data_points = len(adgroup_data)
            if data_points < 10:
                return relative_rmse * 1.5  # Allow 50% more error for small datasets
            elif data_points < 20:
                return relative_rmse * 1.3  # Allow 30% more error for medium datasets
            else:
                return relative_rmse * 1.2  # Allow 20% more error for large datasets
                
        except Exception as e:
            self.logger.warning(f"Error calculating RMSE for ad group: {str(e)}")
            return float('inf')

    def _optimize_spend(self, adgroup, adgroup_data, max_iterations=15):
        """Optimize spend for a single ad group"""
        self.logger.info(f"Starting spend optimization for ad group: {adgroup}")
        
        # Initialize variables
        best_spend = adgroup_data[self.spend_col].mean()  # Start with mean spend instead of min
        best_conversions = adgroup_data[self.target_col].mean()  # Start with mean conversions
        best_rmse = float('inf')
        no_improvement_count = 0
        max_no_improvement = 5
        
        # Calculate spend range based on historical data
        current_spend = adgroup_data[self.spend_col].mean()
        spend_std = adgroup_data[self.spend_col].std()
        
        min_spend = max(self.min_spend, current_spend - 2 * spend_std)
        max_spend = min(self.max_spend, current_spend + 2 * spend_std)
        
        # Generate spend range with safety checks
        try:
            spend_range = np.linspace(
                min_spend,
                max_spend,
                min(int((max_spend - min_spend) / self.step) + 1, 50)  # Reduced max points to 50
            )
            self.logger.info(f"Generated spend range with {len(spend_range)} points")
        except Exception as e:
            self.logger.error(f"Error generating spend range: {str(e)}")
            return current_spend, best_conversions, float('inf')
        
        # Track iterations for debugging
        iteration_count = 0
        
        for spend in spend_range:
            iteration_count += 1
            if iteration_count > max_iterations:
                self.logger.warning(f"Reached maximum iterations ({max_iterations}) for ad group {adgroup}")
                break
                
            try:
                # Predict conversions for this spend level
                spend_df = adgroup_data.copy()
                spend_df[self.spend_col] = spend
                spend_df = self.engineer_features(spend_df)
                
                # Ensure all required features are present
                missing_features = set(self.feature_columns) - set(spend_df.columns)
                if missing_features:
                    self.logger.error(f"Missing features: {missing_features}")
                    continue
                
                # Scale features while maintaining DataFrame structure
                spend_df_scaled = pd.DataFrame(
                    self.scaler.transform(spend_df[self.feature_columns]),
                    columns=self.feature_columns,
                    index=spend_df.index
                )
                
                # Get predictions from both models with error handling
                try:
                    rf_pred = self.models[adgroup]['rf'].predict(spend_df_scaled)
                    gb_pred = self.models[adgroup]['gb'].predict(spend_df_scaled)
                    
                    # Ensure predictions are valid
                    if np.any(np.isnan(rf_pred)) or np.any(np.isnan(gb_pred)):
                        continue
                    
                    # Combine predictions
                    predicted_conversions = (
                        self.models[adgroup]['rf_weight'] * rf_pred +
                        self.models[adgroup]['gb_weight'] * gb_pred
                    )
                    
                    # Ensure predicted conversions are non-negative
                    predicted_conversions = np.maximum(predicted_conversions, 0)
                    
                    # Calculate RMSE
                    current_rmse = np.sqrt(mean_squared_error(
                        adgroup_data[self.target_col],
                        predicted_conversions
                    ))
                    
                    # Update best values if improved
                    if current_rmse < best_rmse and not np.isinf(current_rmse):
                        best_rmse = current_rmse
                        best_spend = spend
                        best_conversions = np.mean(predicted_conversions)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in prediction for spend {spend}: {str(e)}")
                    continue
                
                # Early stopping if no improvement
                if no_improvement_count >= max_no_improvement:
                    self.logger.info(f"Early stopping for ad group {adgroup} after {no_improvement_count} iterations without improvement")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration_count} for ad group {adgroup}: {str(e)}")
                continue
        
        # If no valid solution found, use current spend
        if np.isinf(best_rmse):
            best_spend = current_spend
            best_conversions = adgroup_data[self.target_col].mean()
            best_rmse = float('inf')
        
        self.logger.info(f"Completed optimization for ad group {adgroup}. Best spend: ${best_spend:.2f}, RMSE: {best_rmse:.4f}")
        return best_spend, best_conversions, best_rmse

    def optimize_budget(self, adgroup_data):
        """Optimize budget for a single ad group with RMSE-optimized approach"""
        adgroup_name = adgroup_data['AdGroup'].iloc[0]
        logger.info(f"Starting budget optimization for {adgroup_name}")
        
        if len(adgroup_data) < 3:
            logger.warning(f"Insufficient data points for {adgroup_name}. Skipping.")
            return None
        
        # Calculate current metrics
        current_spend = adgroup_data[self.spend_col].mean()
        current_conversions = adgroup_data[self.target_col].mean()
        
        # Engineer features
        logger.info(f"Engineering features for {adgroup_name}")
        X = self.engineer_features(adgroup_data.copy())
        y = adgroup_data[self.target_col]
        
        # Store feature columns
        self.feature_columns = [col for col in X.columns if col != self.target_col]
        logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")
        
        # Scale features while maintaining DataFrame structure
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X[self.feature_columns]),
            columns=self.feature_columns,
            index=X.index
        )
        
        # Fit ensemble model
        logger.info(f"Fitting ensemble model for {adgroup_name}")
        rf_model, gb_model, rf_weight, gb_weight = self.fit_ensemble_model(X_scaled, y)
        
        # Store models for this ad group
        self.models[adgroup_name] = {
            'rf': rf_model,
            'gb': gb_model,
            'rf_weight': rf_weight,
            'gb_weight': gb_weight
        }
        
        # Optimize spend
        best_spend, best_conversions, best_rmse = self._optimize_spend(adgroup_name, adgroup_data)
        
        # Generate performance plots
        logger.info(f"Generating performance plots for {adgroup_name}")
        y_pred = self.predict_with_ensemble(X_scaled, rf_model, gb_model, rf_weight, gb_weight)
        self.performance_plots[adgroup_name] = self.generate_performance_plots(y, y_pred, adgroup_name)
        
        # Determine confidence level based on relative RMSE
        if best_rmse < 10:
            confidence_level = "High"
        elif best_rmse < 20:
            confidence_level = "Good"
        elif best_rmse < 30:
            confidence_level = "Moderate"
        elif best_rmse < 40:
            confidence_level = "Low"
        else:
            confidence_level = "Poor"
        
        # Generate business justification
        if best_spend > current_spend:
            justification = f"Recommended to increase spend by ${best_spend - current_spend:.2f} to maximize marginal returns"
        elif best_spend < current_spend:
            justification = f"Recommended to decrease spend by ${current_spend - best_spend:.2f} to optimize efficiency"
        else:
            justification = "Current spend level is optimal"
        
        result = {
            'AdGroup': adgroup_name,
            'Current_Spend': current_spend,
            'Current_Conversions': current_conversions,
            'Current_TCPA': current_spend / current_conversions if current_conversions > 0 else float('inf'),
            'Recommended_Spend': best_spend,
            'Expected_Conversions': best_conversions,
            'Expected_TCPA': best_spend / best_conversions if best_conversions > 0 else float('inf'),
            'Relative_RMSE': best_rmse,
            'Confidence': best_conversions,
            'Confidence_Level': confidence_level,
            'Business_Justification': justification
        }
        
        logger.info(f"Completed optimization for {adgroup_name}")
        return result
    
    def optimize_all_adgroups(self):
        """Optimize budget allocation across all ad groups"""
        self.logger.info("Starting optimization for all ad groups")
        
        # Initialize results storage
        results = []
        skipped_adgroups = []
        
        # Get unique ad groups
        adgroups = self.data['AdGroup'].unique()
        total_adgroups = len(adgroups)
        self.logger.info(f"Found {total_adgroups} unique ad groups")
        
        # Limit the number of ad groups to process
        max_adgroups = 50  # Safety limit
        if total_adgroups > max_adgroups:
            self.logger.warning(f"Limiting processing to {max_adgroups} ad groups out of {total_adgroups}")
            adgroups = adgroups[:max_adgroups]
        
        # Process each ad group with timeout
        for i, adgroup in enumerate(adgroups, 1):
            try:
                self.logger.info(f"Processing ad group {i}/{len(adgroups)}: {adgroup}")
                
                # Get data for this ad group
                adgroup_data = self.data[self.data['AdGroup'] == adgroup].copy()
                
                if len(adgroup_data) < 3:
                    self.logger.warning(f"Skipping {adgroup}: insufficient data points ({len(adgroup_data)})")
                    skipped_adgroups.append(adgroup)
                    continue
                
                # Optimize budget for this ad group
                result = self.optimize_budget(adgroup_data)
                
                if result is not None:
                    results.append(result)
                    self.logger.info(f"Successfully optimized {adgroup}")
                else:
                    self.logger.warning(f"Failed to optimize {adgroup}")
                    skipped_adgroups.append(adgroup)
                
            except Exception as e:
                self.logger.error(f"Error processing ad group {adgroup}: {str(e)}")
                skipped_adgroups.append(adgroup)
                continue
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
            self.logger.info(f"Successfully optimized {len(results)} ad groups")
            self.logger.info(f"Skipped {len(skipped_adgroups)} ad groups")
            return results_df
        else:
            self.logger.error("No ad groups were successfully optimized")
            return pd.DataFrame()
