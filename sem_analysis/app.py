import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from global_marginal_return_optimizer import global_marginal_return_optimizer, plot_response_curves_conversions
from global_marginal_return_optimizer_multi_variant import global_marginal_return_optimizer_multi_variant
import os
import base64
from io import BytesIO
import logging

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
model_choice = st.sidebar.radio(
    "Select Optimization Model",
    (
        "Diminishing Returns Model",
        "Multi-Factor Optimization Model"
    ),
    help="Choose which model to use for budget optimization."
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
    st.markdown(f"### Model in Use: **{model_choice}**")
    
    # Run the selected optimization model
    if model_choice == "Diminishing Returns Model":
        results_df = global_marginal_return_optimizer(df, total_budget)
        skipped_adgroups = []
    else:
        results_df, skipped_adgroups = global_marginal_return_optimizer_multi_variant(df, total_budget)
    
    # Show skipped ad groups if any
    if skipped_adgroups:
        st.warning(f"{len(skipped_adgroups)} ad group(s) were skipped due to insufficient or poor model fit.")
        st.expander("See Skipped Ad Groups").write(
            pd.DataFrame(skipped_adgroups)
        )
    
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
                    'Confidence': '{:.2%}'
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