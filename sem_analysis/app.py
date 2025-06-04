import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from global_marginal_return_optimizer import global_marginal_return_optimizer, plot_response_curves_conversions
import os
import base64
from io import BytesIO

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
total_budget = st.sidebar.number_input(
    "Total Weekly Budget ($)",
    min_value=1000,
    max_value=1000000,
    value=40000,
    step=1000
)

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv('sem_analysis/SEM_DATA_TOP100.csv')
    df['week_start'] = pd.to_datetime(df['week_start'])
    return df

try:
    df = load_data()
    
    # Run optimization
    results_df = global_marginal_return_optimizer(df, total_budget)
    
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
        st.dataframe(
            results_df.style.format({
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
        for _, row in results_df.iterrows():
            with st.expander(f"📊 {row['AdGroup']}"):
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

except Exception as e:
    st.error(f"An error occurred: {str(e)}") 