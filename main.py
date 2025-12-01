"""
Main execution script for Loblaw Bio clinical trial analysis.

Usage:
    python main.py --init-db
    python main.py --load-data data/raw/cell-count.csv
    python main.py --info
"""

import argparse
import sqlite3
import yaml
from pathlib import Path
from src.analysis.summary_stats import FrequencyAnalyzer, export_frequency_table
from src.analysis.statistical_tests import ResponseAnalyzer
from src.analysis.filtering import CohortFilter
from src.visualization.plots import TrialVisualizer
import matplotlib.pyplot as plt
from src.database.schema import DatabaseSchema, create_database
from src.database.loader import DataLoader


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def init_database(config: dict) -> None:
    """Initialize database with schema."""
    print("\nInitializing database...")
    
    db_path = config['database']['path']
    populations = config['cell_populations']
    
    conn = create_database(db_path, populations)
    conn.close()
    
    print(f"✓ Database initialized: {db_path}\n")


def load_data(config: dict, csv_path: str = None) -> None:
    """Load data from CSV into database."""
    if csv_path is None:
        csv_path = config['data']['raw_csv']
    
    db_path = config['database']['path']
    
    # Check if database exists
    if not Path(db_path).exists():
        print("Database not found. Initializing...")
        init_database(config)
    
    # Connect and load
    conn = sqlite3.connect(db_path)
    loader = DataLoader(conn)
    loader.load_csv_to_database(csv_path)
    conn.close()


def show_info(config: dict) -> None:
    """Display database information."""
    db_path = config['database']['path']
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run: python main.py --init-db")
        return
    
    conn = sqlite3.connect(db_path)
    schema = DatabaseSchema(db_path)
    info = schema.get_database_info(conn)
    conn.close()
    
    print("\n" + "="*60)
    print("DATABASE INFORMATION")
    print("="*60)
    for key, value in info.items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")

def run_frequency_analysis(config: dict) -> None:
    """Run frequency analysis (Part 2)."""
    db_path = config['database']['path']
    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*60)
    print("FREQUENCY ANALYSIS")
    print("="*60 + "\n")
    
    analyzer = FrequencyAnalyzer(conn)
    
    # Calculate frequencies
    freq_df = analyzer.calculate_frequency_table()
    summary = analyzer.get_summary_statistics()
    
    print("Summary Statistics:")
    print(summary)
    
    # Export to CSV
    output_path = "outputs/frequency_table.csv"
    export_frequency_table(conn, output_path)
    
    conn.close()
    print("\n✓ Frequency analysis complete!\n")


def run_response_analysis(config: dict) -> None:
    """Run response analysis (Part 3)."""
    db_path = config['database']['path']
    params = config['analysis']['part3']
    
    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*60)
    print("RESPONSE ANALYSIS")
    print("="*60 + "\n")
    
    analyzer = ResponseAnalyzer(conn)
    
    # Get data and run analysis
    data = analyzer.get_response_data(
        indication=params['indication'],
        treatment=params['treatment'],
        sample_type=params['sample_type']
    )
    
    results = analyzer.analyze_all_populations(
        indication=params['indication'],
        treatment=params['treatment'],
        sample_type=params['sample_type']
    )
    
    # Print results
    analyzer.print_results_summary(results)
    
    # Save results
    results.to_csv("outputs/response_analysis_results.csv", index=False)
    print("✓ Results saved to: outputs/response_analysis_results.csv")
    
    # Create visualization
    print("\nCreating boxplot...")
    viz = TrialVisualizer()
    fig = viz.create_response_boxplot_with_stats(
        data, 
        results,
        save_path="outputs/response_boxplot.png"
    )
    plt.close()
    
    conn.close()
    print("\n✓ Response analysis complete!\n")


def run_cohort_analysis(config: dict) -> None:
    """Run cohort analysis (Part 4)."""
    db_path = config['database']['path']
    params = config['analysis']['part3']
    
    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*60)
    print("COHORT ANALYSIS")
    print("="*60 + "\n")
    
    cohort = CohortFilter(conn)
    
    # Get baseline samples
    baseline_df = cohort.get_baseline_samples(
        indication=params['indication'],
        treatment=params['treatment'],
        sample_type=params['sample_type']
    )
    
    # Summarize
    summary = cohort.summarize_cohort(baseline_df)
    cohort.print_cohort_summary(summary)
    
    # Export
    baseline_df.to_csv("outputs/baseline_cohort.csv", index=False)
    print("✓ Baseline cohort data saved to: outputs/baseline_cohort.csv")
    
    conn.close()
    print("\n✓ Cohort analysis complete!\n")


def run_all_analyses(config: dict) -> None:
    """Run all analyses (Parts 2, 3, 4)."""
    print("\n" + "="*60)
    print("RUNNING ALL ANALYSES")
    print("="*60 + "\n")
    
    run_frequency_analysis(config)
    run_response_analysis(config)
    run_cohort_analysis(config)
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE")
    print("="*60)
    print("\nGenerated files in outputs/:")
    print("  - frequency_table.csv")
    print("  - response_analysis_results.csv")
    print("  - response_boxplot.png")
    print("  - baseline_cohort.csv")
    print("="*60 + "\n")


def launch_dashboard(config: dict) -> None:
    """Launch Streamlit dashboard."""
    import subprocess
    import sys
    
    print("\nLaunching dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Loblaw Bio Clinical Trial Analysis Pipeline"
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database with schema'
    )
    
    parser.add_argument(
        '--load-data',
        type=str,
        metavar='CSV_PATH',
        help='Load data from CSV file'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Display database information'
    )
    
    parser.add_argument(
        '--frequency-analysis',
        action='store_true',
        help='Run frequency analysis (Part 2)'
    )
    
    parser.add_argument(
        '--response-analysis',
        action='store_true',
        help='Run response analysis (Part 3)'
    )
    
    parser.add_argument(
        '--cohort-analysis',
        action='store_true',
        help='Run cohort analysis (Part 4)'
    )
    
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run all analyses (Parts 2, 3, 4)'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch interactive dashboard'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute commands
    if args.init_db:
        init_database(config)
    
    if args.load_data:
        load_data(config, args.load_data)
    
    if args.info:
        show_info(config)
    
    if args.frequency_analysis:
        run_frequency_analysis(config)
    
    if args.response_analysis:
        run_response_analysis(config)
    
    if args.cohort_analysis:
        run_cohort_analysis(config)
    
    if args.run_all:
        run_all_analyses(config)
    
    if args.dashboard:
        launch_dashboard(config)
    
    # If no arguments, show help
    if not any([args.init_db, args.load_data, args.info, 
                args.frequency_analysis, args.response_analysis, 
                args.cohort_analysis, args.run_all, args.dashboard]):
        parser.print_help()


if __name__ == "__main__":
    main()