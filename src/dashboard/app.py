"""
Interactive Streamlit dashboard for Loblaw Bio clinical trial analysis.

Usage:
    streamlit run src/dashboard/app.py
"""

import os
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.summary_stats import FrequencyAnalyzer
from src.analysis.statistical_tests import ResponseAnalyzer
from src.analysis.filtering import CohortFilter
from src.visualization.plots import TrialVisualizer


# Page configuration
st.set_page_config(
    page_title="Loblaw Bio - Clinical Trial Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_database_connection():
    """Create cached database connection."""
    return sqlite3.connect("data/processed/loblaw_trial.db", check_same_thread=False)


def main():
    """Main dashboard application."""
    
    # Title
    st.title("🧬 Loblaw Bio - Clinical Trial Analysis")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Overview", "📊 Frequency Analysis", "📈 Response Analysis", "👥 Cohort Explorer", "🤖 ML Analysis","ℹ️ Database Info"]
    )
    
    # Get database connection
    conn = get_database_connection()
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview(conn)
    elif page == "📊 Frequency Analysis":
        show_frequency_analysis(conn)
    elif page == "📈 Response Analysis":
        show_response_analysis(conn)
    elif page == "👥 Cohort Explorer":
        show_cohort_explorer(conn)
    elif page == "ℹ️ Database Info":
        show_database_info(conn)
    elif page == "🤖 ML Analysis":
        st.header("🤖 Machine Learning Analysis")
        
        try:
            # Check if files exist
            comparison_path = "outputs/ml_model_comparison.csv"
            importance_path = "outputs/ml_feature_importance.csv"
            image_path = "outputs/ml_analysis_results.png"
            
            if not os.path.exists(comparison_path):
                st.warning("⚠️ ML analysis results not found. Run the analysis first using the main pipeline.")
            else:
                # Load results
                comparison_df = pd.read_csv(comparison_path)
                
                st.markdown("""
                This analysis uses **Random Forest** and **XGBoost** classifiers to predict 
                treatment response based on cell population frequencies.
                """)
                
                # Model comparison - Side by side
                st.subheader("📊 Model Performance Comparison")
                
                # Check what columns we have
                available_cols = comparison_df.columns.tolist()
                st.write(f"Available metrics: {', '.join([c for c in available_cols if c != 'Model'])}")
                
                col1, col2 = st.columns(2)
                
                # Get model data safely
                rf_data = comparison_df[comparison_df['Model'] == 'Random Forest'].iloc[0]
                xgb_data = comparison_df[comparison_df['Model'] == 'XGBoost'].iloc[0]
                
                with col1:
                    st.markdown("### 🌲 Random Forest")
                    
                    # Display metrics that exist
                    if 'Test_ROC_AUC' in available_cols:
                        st.metric("ROC-AUC", f"{rf_data['Test_ROC_AUC']:.4f}")
                    if 'Test_Accuracy' in available_cols:
                        st.metric("Accuracy", f"{rf_data['Test_Accuracy']:.4f}")
                    if 'Test_Precision' in available_cols:
                        st.metric("Precision", f"{rf_data['Test_Precision']:.4f}")
                    if 'Test_Recall' in available_cols:
                        st.metric("Recall", f"{rf_data['Test_Recall']:.4f}")
                    if 'Test_F1' in available_cols:
                        st.metric("F1-Score", f"{rf_data['Test_F1']:.4f}")
                    
                    if 'CV_ROC_AUC_Mean' in available_cols:
                        with st.expander("Cross-Validation Results"):
                            st.write(f"**Mean ROC-AUC:** {rf_data['CV_ROC_AUC_Mean']:.4f}")
                            if 'CV_ROC_AUC_Std' in available_cols:
                                st.write(f"**Std Dev:** {rf_data['CV_ROC_AUC_Std']:.4f}")
                
                with col2:
                    st.markdown("### ⚡ XGBoost")
                    
                    # Display metrics that exist
                    if 'Test_ROC_AUC' in available_cols:
                        st.metric("ROC-AUC", f"{xgb_data['Test_ROC_AUC']:.4f}")
                    if 'Test_Accuracy' in available_cols:
                        st.metric("Accuracy", f"{xgb_data['Test_Accuracy']:.4f}")
                    if 'Test_Precision' in available_cols:
                        st.metric("Precision", f"{xgb_data['Test_Precision']:.4f}")
                    if 'Test_Recall' in available_cols:
                        st.metric("Recall", f"{xgb_data['Test_Recall']:.4f}")
                    if 'Test_F1' in available_cols:
                        st.metric("F1-Score", f"{xgb_data['Test_F1']:.4f}")
                    
                    if 'CV_ROC_AUC_Mean' in available_cols:
                        with st.expander("Cross-Validation Results"):
                            st.write(f"**Mean ROC-AUC:** {xgb_data['CV_ROC_AUC_Mean']:.4f}")
                            if 'CV_ROC_AUC_Std' in available_cols:
                                st.write(f"**Std Dev:** {xgb_data['CV_ROC_AUC_Std']:.4f}")
                
                # Full comparison table
                st.subheader("📊 Detailed Metrics Table")
                st.dataframe(
                    comparison_df.style.format({
                        col: "{:.4f}" for col in comparison_df.columns if col != 'Model'
                    }),
                    use_container_width=True
                )
                
                # Comparison chart
                if len([c for c in available_cols if c.startswith('Test_')]) > 0:
                    st.subheader("📈 Performance Metrics Comparison")
                    
                    metrics_to_compare = [c for c in available_cols if c.startswith('Test_')]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Random Forest',
                        x=[m.replace('Test_', '') for m in metrics_to_compare],
                        y=[rf_data[m] for m in metrics_to_compare],
                        marker_color='blue',
                        text=[f"{rf_data[m]:.3f}" for m in metrics_to_compare],
                        textposition='auto'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='XGBoost',
                        x=[m.replace('Test_', '') for m in metrics_to_compare],
                        y=[xgb_data[m] for m in metrics_to_compare],
                        marker_color='red',
                        text=[f"{xgb_data[m]:.3f}" for m in metrics_to_compare],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        barmode='group',
                        xaxis_title='Metric',
                        yaxis_title='Score',
                        yaxis_range=[0, 1],
                        height=400,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if os.path.exists(importance_path):
                    st.subheader("🔍 Feature Importance Analysis")
                    
                    importance_df = pd.read_csv(importance_path)
                    
                    st.markdown("""
                    Feature importance shows which cell populations are most useful for predicting treatment response.
                    Higher values indicate greater predictive power.
                    """)
                    
                    # Create horizontal bar chart
                    fig2 = go.Figure()
                    
                    # Sort by Random Forest importance
                    importance_df_sorted = importance_df.sort_values('Random_Forest_Importance', ascending=True)
                    
                    fig2.add_trace(go.Bar(
                        name='Random Forest',
                        y=importance_df_sorted['Feature'],
                        x=importance_df_sorted['Random_Forest_Importance'],
                        orientation='h',
                        marker_color='blue',
                        text=importance_df_sorted['Random_Forest_Importance'].round(4),
                        textposition='auto'
                    ))
                    
                    fig2.add_trace(go.Bar(
                        name='XGBoost',
                        y=importance_df_sorted['Feature'],
                        x=importance_df_sorted['XGBoost_Importance'],
                        orientation='h',
                        marker_color='red',
                        text=importance_df_sorted['XGBoost_Importance'].round(4),
                        textposition='auto'
                    ))
                    
                    fig2.update_layout(
                        barmode='group',
                        xaxis_title='Importance Score',
                        yaxis_title='Cell Population',
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Show detailed visualization
                st.subheader("📊 Detailed Analysis Visualizations")
                
                if os.path.exists(image_path):
                    st.image(image_path, caption="ML Analysis Results: ROC Curves, Feature Importance, and Confusion Matrices", use_column_width=True)
                else:
                    st.warning("⚠️ Visualization image not found. Run the analysis to generate visualizations.")
                
                # Download options
                st.subheader("💾 Download Results")
                
                csv_comparison = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Model Comparison",
                    data=csv_comparison,
                    file_name="ml_model_comparison.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error loading ML analysis results: {e}")
            st.exception(e)
            st.info("💡 Try running: `python main.py --run-all` to regenerate the ML analysis.")


def show_overview(conn):
    """Display overview page."""
    st.header("Project Overview")
    
    st.markdown("""
    This dashboard provides interactive analysis of clinical trial data for immune cell populations.
    
    ### Analysis Components:
    
    - **📊 Frequency Analysis**: Calculate and visualize relative frequencies of cell populations
    - **📈 Response Analysis**: Compare responders vs non-responders with statistical testing
    - **👥 Cohort Explorer**: Filter and summarize patient cohorts
    - **ℹ️ Database Info**: View database statistics and metadata
    
    ### Quick Stats:
    """)
    
    # Get quick stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM samples")
    n_samples = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT subject) FROM samples")
    n_subjects = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT project) FROM samples")
    n_projects = cursor.fetchone()[0]
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", f"{n_samples:,}")
    col2.metric("Unique Subjects", f"{n_subjects:,}")
    col3.metric("Projects", n_projects)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between different analyses.")


def show_frequency_analysis(conn):
    """Display frequency analysis page."""
    st.header("📊 Cell Population Frequency Analysis")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Get unique values for filters
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT indication FROM samples ORDER BY indication")
    indications = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT treatment FROM samples ORDER BY treatment")
    treatments = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT sample_type FROM samples ORDER BY sample_type")
    sample_types = [row[0] for row in cursor.fetchall()]
    
    # Filter widgets
    indication = st.sidebar.selectbox("Indication", ["All"] + indications)
    treatment = st.sidebar.selectbox("Treatment", ["All"] + treatments)
    sample_type = st.sidebar.selectbox("Sample Type", ["All"] + sample_types)
    
    # Apply filters
    filters = {}
    if indication != "All":
        filters['indication'] = indication
    if treatment != "All":
        filters['treatment'] = treatment
    if sample_type != "All":
        filters['sample_type'] = sample_type
    
    # Calculate frequencies
    analyzer = FrequencyAnalyzer(conn)
    
    with st.spinner("Calculating frequencies..."):
        freq_df = analyzer.calculate_frequency_table(**filters)
        summary_stats = analyzer.get_summary_statistics(**filters)
    
    # Display results
    st.subheader("Summary Statistics")
    st.dataframe(summary_stats.style.format("{:.2f}"), use_container_width=True)
    
    # Visualization
    st.subheader("Frequency Distribution")
    
    # Create boxplot
    fig = px.box(
        freq_df,
        x='population',
        y='percentage',
        color='population',
        labels={'population': 'Cell Population', 'percentage': 'Relative Frequency (%)'},
        title='Cell Population Frequency Distribution'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download data
    st.subheader("Export Data")
    csv = freq_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Frequency Table (CSV)",
        data=csv,
        file_name="frequency_table.csv",
        mime="text/csv"
    )


def show_response_analysis(conn):
    """Display response analysis page."""
    st.header("📈 Response Analysis: Responders vs Non-Responders")
    
    # Filters
    st.sidebar.subheader("Analysis Parameters")
    indication = st.sidebar.selectbox("Indication", ["melanoma", "carcinoma"], index=0)
    treatment = st.sidebar.selectbox("Treatment", ["miraclib", "phauximab"], index=0)
    sample_type = st.sidebar.selectbox("Sample Type", ["PBMC", "WB"], index=0)
    
    # Run analysis
    analyzer = ResponseAnalyzer(conn)
    
    with st.spinner("Running statistical analysis..."):
        data = analyzer.get_response_data(indication, treatment, sample_type)
        results = analyzer.analyze_all_populations(indication, treatment, sample_type)
    
    # Display sample sizes
    st.subheader("Sample Sizes")
    col1, col2 = st.columns(2)
    col1.metric("Responders", results['n_responders'].iloc[0])
    col2.metric("Non-Responders", results['n_non_responders'].iloc[0])
    
    # Statistical results table
    st.subheader("Statistical Test Results")
    st.markdown("**Test:** Mann-Whitney U with Bonferroni correction")
    st.markdown(f"**Significance threshold:** p < 0.01 (α=0.05/5 populations)")
    
    # Format results for display
    display_results = results[[
        'population', 'median_responders', 'median_non_responders',
        'mean_responders', 'mean_non_responders', 'p_value', 'significant', 'effect_size'
    ]].copy()
    
    display_results.columns = [
        'Population', 'Median Resp.', 'Median Non-Resp.',
        'Mean Resp.', 'Mean Non-Resp.', 'P-value', 'Significant', 'Effect Size'
    ]
    
    # Apply styling
    def highlight_significant(row):
        if row['Significant']:
            return ['background-color: #90EE90'] * len(row)
        return [''] * len(row)
    
    styled_results = display_results.style.format({
        'Median Resp.': '{:.2f}%',
        'Median Non-Resp.': '{:.2f}%',
        'Mean Resp.': '{:.2f}%',
        'Mean Non-Resp.': '{:.2f}%',
        'P-value': '{:.4f}',
        'Effect Size': '{:.3f}'
    }).apply(highlight_significant, axis=1)
    
    st.dataframe(styled_results, use_container_width=True)
    
    # Boxplot
    st.subheader("Frequency Distribution by Response")
    
    viz = TrialVisualizer()
    fig = viz.create_interactive_boxplot(data, results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.subheader("Interpretation")
    sig_pops = results[results['significant']]['population'].tolist()
    if sig_pops:
        st.success(f"✅ Significant differences found in: {', '.join(sig_pops)}")
    else:
        st.info("ℹ️ No statistically significant differences detected after multiple testing correction.")
    
    # Download
    st.subheader("Export Results")
    csv = results.to_csv(index=False)
    st.download_button(
        label="📥 Download Statistical Results (CSV)",
        data=csv,
        file_name=f"response_analysis_{indication}_{treatment}.csv",
        mime="text/csv"
    )


def show_cohort_explorer(conn):
    """Display cohort explorer page."""
    st.header("👥 Cohort Explorer")
    
    # Filters
    st.sidebar.subheader("Filter Criteria")
    
    # Get options
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT indication FROM samples ORDER BY indication")
    indications = ["All"] + [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT treatment FROM samples ORDER BY treatment")
    treatments = ["All"] + [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT sample_type FROM samples ORDER BY sample_type")
    sample_types = ["All"] + [row[0] for row in cursor.fetchall()]
    
    # Filter widgets
    indication = st.sidebar.selectbox("Indication", indications)
    treatment = st.sidebar.selectbox("Treatment", treatments)
    sample_type = st.sidebar.selectbox("Sample Type", sample_types)
    response = st.sidebar.selectbox("Response", ["All", "yes", "no", "not_applicable"])
    gender = st.sidebar.selectbox("Gender", ["All", "M", "F"])
    timepoint = st.sidebar.number_input("Timepoint", min_value=0, max_value=14, value=0, step=7)
    
    # Build filters
    filters = {}
    if indication != "All":
        filters['indication'] = indication
    if treatment != "All":
        filters['treatment'] = treatment
    if sample_type != "All":
        filters['sample_type'] = sample_type
    if response != "All":
        filters['response'] = response
    if gender != "All":
        filters['gender'] = gender
    filters['timepoint'] = timepoint
    
    # Query data
    cohort = CohortFilter(conn)
    filtered_data = cohort.filter_by_criteria(**filters)
    
    # Display summary
    st.subheader("Cohort Summary")
    
    if len(filtered_data) == 0:
        st.warning("No samples match the selected criteria.")
        return
    
    summary = cohort.summarize_cohort(filtered_data)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", summary['total_samples'])
    col2.metric("Unique Subjects", summary['unique_subjects'])
    col3.metric("Age (Mean)", f"{summary['age_mean']:.1f}")
    col4.metric("Age Range", f"{summary['age_range'][0]}-{summary['age_range'][1]}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Response Distribution")
        response_data = pd.DataFrame({
            'Response': ['Responders', 'Non-Responders', 'Not Applicable'],
            'Count': [summary['responders'], summary['non_responders'], summary['not_applicable']]
        })
        fig = px.pie(response_data, values='Count', names='Response', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Gender Distribution")
        gender_data = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Count': [summary['males'], summary['females']]
        })
        fig = px.bar(gender_data, x='Gender', y='Count', color='Gender')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Filtered Samples")
    st.dataframe(filtered_data, use_container_width=True, height=400)
    
    # Download
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_cohort.csv",
        mime="text/csv"
    )


def show_database_info(conn):
    """Display database information page."""
    st.header("ℹ️ Database Information")
    
    cursor = conn.cursor()
    
    # Overview
    st.subheader("Database Overview")
    
    cursor.execute("SELECT COUNT(*) FROM samples")
    n_samples = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT subject) FROM samples")
    n_subjects = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT project) FROM samples")
    n_projects = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cell_counts")
    n_measurements = cursor.fetchone()[0]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{n_samples:,}")
    col2.metric("Unique Subjects", f"{n_subjects:,}")
    col3.metric("Projects", n_projects)
    col4.metric("Measurements", f"{n_measurements:,}")
    
    # Breakdowns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Indication Distribution")
        cursor.execute("SELECT indication, COUNT(*) as count FROM samples GROUP BY indication")
        indication_data = pd.DataFrame(cursor.fetchall(), columns=['Indication', 'Count'])
        fig = px.bar(indication_data, x='Indication', y='Count', color='Indication')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Treatment Distribution")
        cursor.execute("SELECT treatment, COUNT(*) as count FROM samples GROUP BY treatment")
        treatment_data = pd.DataFrame(cursor.fetchall(), columns=['Treatment', 'Count'])
        fig = px.bar(treatment_data, x='Treatment', y='Count', color='Treatment')
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample types
    st.subheader("Sample Type Distribution")
    cursor.execute("SELECT sample_type, COUNT(*) as count FROM samples GROUP BY sample_type")
    sample_type_data = pd.DataFrame(cursor.fetchall(), columns=['Sample Type', 'Count'])
    fig = px.pie(sample_type_data, values='Count', names='Sample Type', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    main()