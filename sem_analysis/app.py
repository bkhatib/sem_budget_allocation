import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from global_marginal_return_optimizer import global_marginal_return_optimizer, plot_response_curves_conversions
from global_marginal_return_optimizer_multi_variant import global_marginal_return_optimizer_multi_variant
from global_marginal_return_RMSE_optimized import GlobalMarginalReturnOptimizerRMSE
import os
import base64
from io import BytesIO
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="SEM Budget Optimization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 SEM Budget Optimization Dashboard")
st.markdown("""
This dashboard helps you understand and optimize your SEM budget allocation across different ad groups.
Use the controls below to adjust parameters and view optimization results.
""")

# Sidebar controls
st.sidebar.header("Optimization Parameters")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Basic Model", "Multi-Factor Model", "RMSE-Optimized Model"],
    help="Choose the model type for budget optimization"
)

# File uploader
st.sidebar.subheader("Upload SEM Data CSV")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file with the same format as the default. If not provided, the default file will be used."
)

total_budget = st.sidebar.number_input(
    "Total Weekly Budget ($)",
    min_value=1000,
    max_value=1000000,
    value=40000,
    step=1000
)

# Add RMSE-optimized model parameters
if model_type == "RMSE-Optimized Model":
    st.sidebar.subheader("RMSE-Optimized Model Parameters")
    min_spend = st.sidebar.number_input(
        "Minimum Spend ($)",
        min_value=0.1,
        max_value=1000,
        value=0.1,
        step=0.1,
        help="Minimum spend per ad group"
    )
    max_spend = st.sidebar.number_input(
        "Maximum Spend ($)",
        min_value=1000,
        max_value=100000,
        value=1000,
        step=100,
        help="Maximum spend per ad group"
    )
    step = st.sidebar.number_input(
        "Optimization Step Size ($)",
        min_value=0.1,
        max_value=100,
        value=0.1,
        step=0.1,
        help="Step size for optimization"
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence level for recommendations"
    )

# Load and process data
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('sem_analysis/SEM_DATA_TOP100.csv')
        df['week_start'] = pd.to_datetime(df['week_start'])
        st.sidebar.success(f"Loaded {len(df)} rows of data")
        st.sidebar.info(f"Number of unique ad groups: {df['AdGroup'].nunique()}")
        st.sidebar.info(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        return None

try:
    df = load_data(uploaded_file)
    
    # Show which model is active
    st.markdown(f"### Model in Use: **{model_type}**")
    
    # Initialize the appropriate model
    if model_type == "Basic Model":
        results_df, adgroup_rmses = global_marginal_return_optimizer(df, total_budget)
        skipped_adgroups = []
    elif model_type == "Multi-Factor Model":
        results_df, skipped_adgroups, adgroup_rmses = global_marginal_return_optimizer_multi_variant(df, total_budget)
    else:  # RMSE-Optimized Model
        optimizer = GlobalMarginalReturnOptimizerRMSE(
            data=df,
            min_spend=min_spend,
            max_spend=max_spend,
            step=step,
            confidence_threshold=confidence_threshold
        )
        results_df = optimizer.optimize_all_adgroups()
        skipped_adgroups = []
        adgroup_rmses = []  # We'll calculate this from the results_df
    
    # Add model description
    if model_type == "Basic Model":
        st.markdown("""
        ### Basic Model
        This model uses a simple logarithmic function to model the relationship between spend and conversions.
        It optimizes budget allocation based on marginal returns, considering only spend and conversion data.
        """)
    elif model_type == "Multi-Factor Model":
        st.markdown("""
        ### Multi-Factor Model
        This model incorporates CTR and CVR metrics to provide a more comprehensive view of ad performance.
        It considers the relationships between spend, impressions, clicks, and conversions to make more informed budget decisions.
        """)
    else:  # RMSE-Optimized Model
        st.markdown("""
        ### RMSE-Optimized Model
        This model uses advanced optimization techniques to minimize prediction error (RMSE) while maximizing conversion potential.
        It includes CTR and CVR metrics, plus additional features for enhanced accuracy.
        """)
    
    # Calculate overall RMSE (weighted by number of data points per ad group)
    if adgroup_rmses:
        total_n = sum(r['n'] for r in adgroup_rmses)
        overall_rmse = np.sqrt(
            sum((r['RMSE']**2) * r['n'] for r in adgroup_rmses) / total_n
        ) if total_n > 0 else np.nan
    else:
        overall_rmse = np.nan
    
    # Show overall RMSE at the top
    st.metric("Overall Model RMSE (conversions)", f"{overall_rmse:.2f}" if not np.isnan(overall_rmse) else "N/A")
    st.info("""
    **RMSE (Root Mean Squared Error) Guide:**
    - **Excellent (0–1):** Very high accuracy
    - **Good (1–3):** Reliable for most decisions
    - **Moderate (3–6):** Use with some caution
    - **Low (6–10):** Review recommendations carefully
    - **Poor (>10):** Use only as a rough guide
    
    RMSE tells you, on average, how far off the model's predictions are from actual conversions. Lower is better.
    """)
    
    # Show skipped ad groups section always
    with st.expander("See Skipped Ad Groups", expanded=False):
        if skipped_adgroups:
            st.warning(f"{len(skipped_adgroups)} ad group(s) were skipped due to insufficient or poor model fit.")
            st.write(pd.DataFrame(skipped_adgroups))
        else:
            st.success("All ad groups were successfully modeled.")
    
    if df is not None:
        # Display data summary
        st.sidebar.subheader("Data Summary")
        st.sidebar.write(f"Total Spend: ${df['Spend'].sum():,.2f}")
        st.sidebar.write(f"Total Conversions: {df['Conversions'].sum():,.0f}")
        st.sidebar.write(f"Average TCPA: ${(df['Spend'].sum() / df['Conversions'].sum()):,.2f}")
        
        # Run optimization
        st.info("Running optimization...")
        
        if not results_df.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Current Spend",
                    f"${results_df['Current_Spend'].sum():,.0f}"
                )
            
            with col2:
                st.metric(
                    "Total Recommended Spend",
                    f"${results_df['Recommended_Spend'].sum():,.0f}"
                )
            
            with col3:
                st.metric(
                    "Expected Conversion Increase",
                    f"{(results_df['Expected_Conversions'].sum() - results_df['Current_Conversions'].sum()):,.1f}"
                )
            
            with col4:
                st.metric(
                    "Average TCPA Improvement",
                    f"${(results_df['Current_TCPA'].mean() - results_df['Expected_TCPA'].mean()):,.2f}"
                )

            # Budget Allocation Chart
            st.subheader("Budget Allocation Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Spend',
                x=results_df['AdGroup'],
                y=results_df['Current_Spend'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Recommended Spend',
                x=results_df['AdGroup'],
                y=results_df['Recommended_Spend'],
                marker_color='darkblue'
            ))
            fig.update_layout(
                barmode='group',
                xaxis_title="Ad Group",
                yaxis_title="Spend ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # TCPA Comparison
            st.subheader("TCPA Comparison")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current TCPA',
                x=results_df['AdGroup'],
                y=results_df['Current_TCPA'],
                marker_color='lightgreen'
            ))
            fig.add_trace(go.Bar(
                name='Expected TCPA',
                x=results_df['AdGroup'],
                y=results_df['Expected_TCPA'],
                marker_color='darkgreen'
            ))
            fig.update_layout(
                barmode='group',
                xaxis_title="Ad Group",
                yaxis_title="TCPA ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed Results Table
            st.subheader("Detailed Optimization Results")
            
            # Add confidence level filter
            confidence_levels = results_df['Confidence_Level'].unique()
            selected_confidence = st.multiselect(
                "Filter by Confidence Level",
                options=confidence_levels,
                default=confidence_levels
            )
            
            # Filter the dataframe based on selected confidence levels
            filtered_df = results_df[results_df['Confidence_Level'].isin(selected_confidence)]
            
            # Display confidence level distribution
            col1, col2 = st.columns(2)
            with col1:
                confidence_counts = results_df['Confidence_Level'].value_counts()
                st.write("Confidence Level Distribution:")
                for level, count in confidence_counts.items():
                    st.write(f"- {level}: {count} ad groups")
            
                # Calculate spend distribution by confidence level (current)
                spend_by_confidence = results_df.groupby('Confidence_Level')['Current_Spend'].sum()
                total_spend = spend_by_confidence.sum()
                st.write("\nCurrent Spend Distribution by Confidence:")
                for level, spend in spend_by_confidence.items():
                    percentage = (spend / total_spend) * 100
                    st.write(f"- {level}: ${spend:,.2f} ({percentage:.1f}% of total spend)")
                
                # Calculate recommended spend distribution by confidence level
                rec_spend_by_confidence = results_df.groupby('Confidence_Level')['Recommended_Spend'].sum()
                total_rec_spend = rec_spend_by_confidence.sum()
                st.write("\nRecommended Spend Distribution by Confidence:")
                for level, spend in rec_spend_by_confidence.items():
                    percentage = (spend / total_rec_spend) * 100 if total_rec_spend > 0 else 0
                    st.write(f"- {level}: ${spend:,.2f} ({percentage:.1f}% of recommended budget)")

            with col2:
                # Calculate average confidence by level
                avg_confidence = results_df.groupby('Confidence_Level')['Confidence'].mean()
                st.write("Average R² by Confidence Level:")
                for level, avg in avg_confidence.items():
                    st.write(f"- {level}: {avg:.2%}")
            
            # Display the filtered dataframe
            st.dataframe(
                filtered_df.style.format({
                    'Current_Spend': '${:,.2f}',
                    'Recommended_Spend': '${:,.2f}',
                    'Current_TCPA': '${:,.2f}',
                    'Expected_TCPA': '${:,.2f}',
                    'Marginal_Conversion_per_Dollar': '{:.4f}',
                    'Confidence': '{:.2%}',
                    'RMSE': '{:.2f}',
                    'MAPE': '{:.1f}%',
                    'NRMSE': '{:.1f}%',
                    'Stability': '{:.2%}',
                    'Direction_Accuracy': '{:.2%}',
                    'Spend_Efficiency': '{:.4f}',
                    'ROI_Projection': '{:.1f}%',
                    'Break_Even_Point': '{:.1f}',
                    'CTR': '{:.2%}',
                    'CVR': '{:.2%}',
                    'CTR_Impact': '{:.2%}',
                    'CVR_Impact': '{:.2%}'
                }),
                use_container_width=True
            )

            # Model Performance Visualization
            if model_type == "RMSE-Optimized Model":
                st.subheader("Model Performance Visualization")
                selected_adgroup = st.selectbox(
                    "Select Ad Group to View Performance",
                    options=filtered_df['AdGroup'].unique()
                )
                
                if selected_adgroup in optimizer.performance_plots:
                    actual_vs_pred, residuals = optimizer.performance_plots[selected_adgroup]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(actual_vs_pred, use_container_width=True)
                    with col2:
                        st.plotly_chart(residuals, use_container_width=True)
                    
                    # Display detailed metrics for selected ad group
                    adgroup_metrics = filtered_df[filtered_df['AdGroup'] == selected_adgroup].iloc[0]
                    
                    st.subheader("Detailed Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R² Score", f"{adgroup_metrics['R2']:.2%}")
                        st.metric("RMSE", f"{adgroup_metrics['RMSE']:.2f}")
                        st.metric("MAPE", f"{adgroup_metrics['MAPE']:.1f}%")
                    
                    with col2:
                        st.metric("NRMSE", f"{adgroup_metrics['NRMSE']:.1f}%")
                        st.metric("Stability", f"{adgroup_metrics['Stability']:.2%}")
                        st.metric("Direction Accuracy", f"{adgroup_metrics['Direction_Accuracy']:.2%}")
                    
                    with col3:
                        st.metric("Spend Efficiency", f"{adgroup_metrics['Spend_Efficiency']:.4f}")
                        st.metric("ROI Projection", f"{adgroup_metrics['ROI_Projection']:.1f}%")
                        st.metric("Break-Even Point", f"{adgroup_metrics['Break_Even_Point']:.1f}")
                    
                    with col4:
                        if 'CTR' in adgroup_metrics:
                            st.metric("CTR", f"{adgroup_metrics['CTR']:.2%}")
                        if 'CVR' in adgroup_metrics:
                            st.metric("CVR", f"{adgroup_metrics['CVR']:.2%}")
                        if 'CTR_Impact' in adgroup_metrics:
                            st.metric("CTR Impact", f"{adgroup_metrics['CTR_Impact']:.2%}")
                        if 'CVR_Impact' in adgroup_metrics:
                            st.metric("CVR Impact", f"{adgroup_metrics['CVR_Impact']:.2%}")

            # Business Justifications
            st.subheader("Business Justifications")
            for _, row in filtered_df.iterrows():
                with st.expander(f"📊 {row['AdGroup']} ({row['Confidence_Level']})"):
                    st.markdown(row['Business_Justification'])
                    if model_type == "RMSE-Optimized Model":
                        metrics_text = """
                        **Model Performance Metrics:**
                        - R² Score: {:.2%}
                        - RMSE: {:.2f}
                        - MAPE: {:.1f}%
                        - NRMSE: {:.1f}%
                        - Stability: {:.2%}
                        - Direction Accuracy: {:.2%}
                        
                        **Business Impact:**
                        - Spend Efficiency: {:.4f}
                        - ROI Projection: {:.1f}%
                        - Break-Even Point: {:.1f}
                        """.format(
                            row['R2'], row['RMSE'], row['MAPE'], row['NRMSE'],
                            row['Stability'], row['Direction_Accuracy'],
                            row['Spend_Efficiency'], row['ROI_Projection'],
                            row['Break_Even_Point']
                        )
                        
                        # Add CTR and CVR metrics if available
                        if 'CTR' in row:
                            metrics_text += f"\n**Engagement Metrics:**\n- CTR: {row['CTR']:.2%}"
                        if 'CVR' in row:
                            metrics_text += f"\n- CVR: {row['CVR']:.2%}"
                        if 'CTR_Impact' in row:
                            metrics_text += f"\n- CTR Impact: {row['CTR_Impact']:.2%}"
                        if 'CVR_Impact' in row:
                            metrics_text += f"\n- CVR Impact: {row['CVR_Impact']:.2%}"
                        
                        st.markdown(metrics_text)

            # Download buttons
            st.sidebar.markdown("---")
            st.sidebar.subheader("Download Results")
            
            # CSV Download
            csv = results_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="optimization_results.csv",
                mime="text/csv"
            )
            
            # Generate and save response curves
            plot_response_curves_conversions(df, results_df)
            
            # Create zip file of response curves
            import zipfile
            import io
            
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                for filename in os.listdir('response_curves_conversions'):
                    if filename.endswith('.png'):
                        zf.write(
                            os.path.join('response_curves_conversions', filename),
                            filename
                        )
            
            st.sidebar.download_button(
                label="Download Response Curves",
                data=memory_file.getvalue(),
                file_name="response_curves.zip",
                mime="application/zip"
            )

        else:
            st.error("No valid optimization results. Please check your data and parameters.")
            # Add debugging information
            st.subheader("Debugging Information")
            st.write("Data Summary:")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Number of ad groups: {df['AdGroup'].nunique()}")
            st.write(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
            st.write(f"Total spend: ${df['Spend'].sum():,.2f}")
            st.write(f"Total conversions: {df['Conversions'].sum():,.0f}")
            
            # Show sample of data
            st.write("Sample of data:")
            st.dataframe(df.head())

    else:
        st.error("Failed to load data. Please check if the data file exists and is properly formatted.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.code(traceback.format_exc()) 