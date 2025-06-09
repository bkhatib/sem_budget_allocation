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
model_type = st.radio(
    "Select Model Type",
    ["Global Marginal Return", "Multi-Factor Model"],
    horizontal=True
)

# Add Model Documentation
with st.expander("📚 Model Documentation - Click to understand the models"):
    if model_type == "Global Marginal Return":
        st.markdown("""
        ### Global Marginal Return Model Documentation

        #### Purpose
        This model optimizes SEM budget allocation across ad groups by analyzing the relationship between spend and conversions, focusing on marginal returns.

        #### Core Concepts
        1. **Marginal Return Concept**
           - Measures the additional conversions gained from each additional dollar spent
           - Helps identify the point of diminishing returns
           - Enables optimal budget distribution across ad groups

        2. **Model Logic**
           - Fits a logarithmic function to each ad group's spend vs. conversion data
           - Formula: Conversions = a * ln(Spend) + b
           - Calculates marginal conversion rate (derivative of the function)
           - Uses this to determine optimal spend levels

        3. **Key Metrics**
           - **R² (R-squared)**: Measures how well the model fits the data (0-1)
             - 0.7-1.0: Excellent fit
             - 0.5-0.7: Good fit
             - 0.3-0.5: Moderate fit
             - <0.3: Poor fit
           - **RMSE**: Measures prediction accuracy in conversion units
           - **Marginal Conversion Rate**: Additional conversions per dollar spent



      
        """)
    else:
        st.markdown("""
        ### Multi-Factor Model Documentation

        #### Purpose
        This advanced model extends the basic marginal return concept by incorporating multiple variables that affect conversion performance.

        #### Core Concepts
        1. **Multi-Variant Analysis**
           - Considers multiple factors affecting conversions
           - Includes spend, CTR, CVR, and their interactions
           - Provides more nuanced optimization

        2. **Model Logic**
           - Uses multiple regression analysis
           - Incorporates interaction terms between variables
           - Provides more sophisticated marginal return calculations
           - Better handles complex relationships between variables

        3. **Key Metrics**
           - **R² (R-squared)**: Measures overall model fit (0-1)
             - 0.7-1.0: Excellent fit
             - 0.5-0.7: Good fit
             - 0.3-0.5: Moderate fit
             - <0.3: Poor fit
           - **RMSE**: Measures prediction accuracy in conversion units
           - **Variable Importance**: Shows which factors most influence conversions
           - **Interaction Effects**: Reveals how variables work together

        4. **Algorithm Steps**
           1. Multi-variable data preparation
           2. Feature engineering and interaction terms
           3. Multiple regression analysis
           4. Marginal return calculation for each variable
           5. Comprehensive budget optimization
           6. Advanced confidence scoring

 
        """)

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

# Load and process data
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Use absolute path to ensure file is found
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, 'SEM_DATA_TOP100.csv')
            df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['AdGroup', 'week_start', 'Spend', 'Conversions', 'Clicks', 'Impressions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.sidebar.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Convert and validate data types
        df['week_start'] = pd.to_datetime(df['week_start'])
        df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
        df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
        df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
        df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna(subset=['Spend', 'Conversions', 'Clicks', 'Impressions'])
        
        if len(df) == 0:
            st.sidebar.error("No valid data after cleaning")
            return None
        
        st.sidebar.success(f"Loaded {len(df)} rows of data")
        st.sidebar.info(f"Number of unique ad groups: {df['AdGroup'].nunique()}")
        st.sidebar.info(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}", exc_info=True)
        return None

try:
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("Failed to load data. Please check the file format and try again.")
        st.stop()
    
    # Show which model is active
    st.markdown(f"### Model in Use: **{model_type}**")
    
    # Initialize the appropriate model
    if model_type == "Global Marginal Return":
        results_df, adgroup_rmses = global_marginal_return_optimizer(df, total_budget)
        skipped_adgroups = []
    elif model_type == "Multi-Factor Model":
        results_df, skipped_adgroups, adgroup_rmses = global_marginal_return_optimizer_multi_variant(df, total_budget)
    else:  # Super Model (Coming Soon)
        st.warning("The Super Model is currently under development and will be available soon.")
        st.stop()
    
    # Add model description
    if model_type == "Global Marginal Return":
        st.markdown("""
        ### Global Marginal Return
        This model uses a simple logarithmic function to model the relationship between spend and conversions.
        It optimizes budget allocation based on marginal returns, considering only spend and conversion data.
        """)
    elif model_type == "Multi-Factor Model":
        st.markdown("""
        ### Multi-Factor Model
        This model incorporates multiple variables that affect conversion performance.
        It adjusts budget allocation based on prediction confidence, providing more conservative and balanced recommendations.
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
                    'CVR_Impact': '{:.2%}',
                    'Marginal_Conversion_per_Dollar': '{:.4f}'
                }),
                use_container_width=True
            )

            # Business Justifications
            st.subheader("Business Justifications")
            for _, row in filtered_df.iterrows():
                with st.expander(f"📊 {row['AdGroup']} ({row['Confidence_Level']})"):
                    st.markdown(row['Business_Justification'])

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