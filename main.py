import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_dataset, standardize_missing_values, smart_sample
from data_analyzer import DataAnalyzer
from data_preprocessor import DataPreprocessor
import logging
from tqdm import tqdm
import base64
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import time
import sys

# Load environment variables from .env file if exists
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Console output formatting helpers
def print_header(text):
    """Print a header with decoration"""
    width = min(100, max(80, len(text) + 20))
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width + "\n")


def print_subheader(text):
    """Print a subheader with decoration"""
    print(f"\n--- {text} ---")


def print_step(step_num, total_steps, description):
    """Print a step indicator"""
    print(f"[Step {step_num}/{total_steps}] {description}")


def print_analysis(text, indent=0):
    """Print analysis information with indentation"""
    indent_str = " " * indent
    print(f"{indent_str}→ {text}")


def print_thinking(text):
    """Print thinking output with animation"""
    print("\nThinking", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print("\n" + text)


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar"""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()


def generate_html_report(original_df, processed_df, analysis_results, output_path):
    """
    Generate an HTML report of the preprocessing steps and results

    Args:
        original_df: Original DataFrame
        processed_df: Processed DataFrame
        analysis_results: Analysis results
        output_path: Path to save HTML report
    """
    print_subheader(f"Generating HTML report at {output_path}")

    html = ["<!DOCTYPE html>", "<html>", "<head>",
            "<title>Data Preprocessing Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".container { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".comparison { display: flex; flex-wrap: wrap; }",
            ".comparison div { flex: 1; min-width: 300px; }",
            "</style>",
            "</head>", "<body>"]

    # Add header
    html.append("<h1>Data Preprocessing Report</h1>")
    html.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

    # Dataset info
    html.append("<div class='container'>")
    html.append(f"<h2>Dataset Overview</h2>")
    html.append(f"<p>Original shape: {original_df.shape[0]} rows, {original_df.shape[1]} columns</p>")
    html.append(f"<p>Processed shape: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns</p>")
    html.append("</div>")

    # Column changes
    original_cols = set(original_df.columns)
    processed_cols = set(processed_df.columns)
    new_cols = processed_cols - original_cols
    removed_cols = original_cols - processed_cols

    html.append("<div class='container'>")
    html.append("<h2>Column Changes</h2>")
    if new_cols:
        html.append(f"<p>Added columns ({len(new_cols)}): {', '.join(new_cols)}</p>")
    if removed_cols:
        html.append(f"<p>Removed columns ({len(removed_cols)}): {', '.join(removed_cols)}</p>")
    html.append("</div>")

    # Analysis results
    html.append("<div class='container'>")
    html.append("<h2>Analysis Results</h2>")
    html.append("<h3>Column-specific Analysis</h3>")
    html.append("<table>")
    html.append(
        "<tr><th>Column</th><th>Data Type</th><th>Missing Values</th><th>Outliers</th><th>Transformations</th></tr>")

    for column, analysis in analysis_results.get('column_analyses', {}).items():
        html.append("<tr>")
        html.append(f"<td>{column}</td>")
        html.append(f"<td>{analysis.get('data_type', '')}</td>")
        html.append(f"<td>{analysis.get('missing_values', '')}</td>")
        html.append(f"<td>{analysis.get('outliers', '')}</td>")
        html.append(f"<td>{analysis.get('transformations', '')}</td>")
        html.append("</tr>")

    html.append("</table>")

    # Overall recommendations
    html.append("<h3>Overall Recommendations</h3>")
    html.append("<ol>")
    for rec in analysis_results.get('overall_recommendations', []):
        html.append(f"<li>{rec}</li>")
    html.append("</ol>")

    # Potential issues
    if analysis_results.get('potential_issues'):
        html.append("<h3>Potential Issues</h3>")
        html.append("<ul>")
        for issue in analysis_results.get('potential_issues', []):
            html.append(f"<li>{issue}</li>")
        html.append("</ul>")

    html.append("</div>")

    # Create plots for numeric columns comparison
    html.append("<div class='container'>")
    html.append("<h2>Numeric Features Comparison</h2>")
    html.append("<div class='comparison'>")

    # Get common numeric columns
    numeric_cols = [
        col for col in original_df.columns
        if col in processed_df.columns and pd.api.types.is_numeric_dtype(original_df[col])
    ]

    print_analysis("Creating comparison visualizations for numeric columns")

    for col in numeric_cols[:5]:  # Limit to 5 plots for brevity
        try:
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            # Original data histogram
            sns.histplot(original_df[col].dropna(), ax=ax1, kde=True)
            ax1.set_title(f"Original: {col}")

            # Processed data histogram
            sns.histplot(processed_df[col].dropna(), ax=ax2, kde=True)
            ax2.set_title(f"Processed: {col}")

            # Save plot to embed in HTML
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            # Embed in HTML
            html.append(f"<div><img src='data:image/png;base64,{img_str}' width='100%'></div>")
            print_analysis(f"Created comparison visualization for column: {col}", indent=2)
        except Exception as e:
            logger.warning(f"Error creating comparison plot for {col}: {str(e)}")
            print_analysis(f"Could not create visualization for column: {col} - {str(e)}", indent=2)

    html.append("</div>")  # Close comparison div
    html.append("</div>")  # Close container

    # Missing values visualization
    html.append("<div class='container'>")
    html.append("<h2>Missing Values Before & After</h2>")

    print_analysis("Creating missing values visualization")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot missing values in original data
        missing_original = original_df.isna().sum().sort_values(ascending=False)
        if not missing_original.empty:
            sns.barplot(x=missing_original.index[:15], y=missing_original.values[:15], ax=ax1)
            ax1.set_title("Missing Values (Original)")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
            ax1.set_ylabel("Count")
        else:
            ax1.set_title("No Missing Values (Original)")

        # Plot missing values in processed data
        missing_processed = processed_df.isna().sum().sort_values(ascending=False)
        if not missing_processed.empty:
            sns.barplot(x=missing_processed.index[:15], y=missing_processed.values[:15], ax=ax2)
            ax2.set_title("Missing Values (Processed)")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
            ax2.set_ylabel("Count")
        else:
            ax2.set_title("No Missing Values (Processed)")

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        html.append(f"<img src='data:image/png;base64,{img_str}' width='100%'>")
        print_analysis("Missing values visualization created successfully", indent=2)
    except Exception as e:
        logger.warning(f"Error creating missing values visualization: {str(e)}")
        print_analysis(f"Could not create missing values visualization: {str(e)}", indent=2)

    html.append("</div>")  # Close container

    # Close HTML
    html.append("</body></html>")

    # Save HTML report
    with open(output_path, 'w') as f:
        f.write("\n".join(html))

    print_analysis(f"HTML report saved to {output_path}", indent=2)


def process_with_progress(df, analysis_results):
    """
    Process dataset with progress tracking

    Args:
        df: Original DataFrame
        analysis_results: Analysis results

    Returns:
        Processed DataFrame
    """
    print_header("DETAILED PREPROCESSING")

    preprocessor = DataPreprocessor(analysis_results)
    column_analyses = analysis_results.get('column_analyses', {})
    processed_df = df.copy()

    total_columns = len([c for c in column_analyses.keys() if c in processed_df.columns])
    print_analysis(f"Starting detailed preprocessing for {total_columns} columns")

    # Process each column with visible steps
    for idx, (column, analysis) in enumerate(column_analyses.items(), 1):
        if column not in processed_df.columns:
            continue

        print_subheader(f"Processing column: {column} ({idx}/{total_columns})")
        print_analysis(f"Original dtype: {df[column].dtype}")
        print_analysis(f"Missing values: {df[column].isna().sum()} ({df[column].isna().mean() * 100:.2f}%)")

        if pd.api.types.is_numeric_dtype(df[column]):
            print_analysis(f"Range: {df[column].min()} to {df[column].max()}")
            print_analysis(f"Mean: {df[column].mean():.2f}, Median: {df[column].median()}")
        elif pd.api.types.is_object_dtype(df[column]):
            print_analysis(f"Unique values: {df[column].nunique()}")
            if df[column].nunique() < 10:
                value_counts = df[column].value_counts().head(5).to_dict()
                print_analysis(f"Top values: {value_counts}")

        # Show the analysis recommendations
        print_analysis("GPT-5 Recommendations:", indent=2)
        if 'data_type' in analysis:
            print_analysis(f"Data type: {analysis['data_type']}", indent=4)
        if 'missing_values' in analysis:
            print_analysis(f"Missing values: {analysis['missing_values']}", indent=4)
        if 'outliers' in analysis:
            print_analysis(f"Outliers: {analysis['outliers']}", indent=4)
        if 'transformations' in analysis:
            print_analysis(f"Transformations: {analysis['transformations']}", indent=4)

        # Data type conversion
        print_analysis("Converting data type...", indent=2)
        before_type = processed_df[column].dtype
        processed_df = preprocessor._convert_data_types(processed_df, column, analysis)
        after_type = processed_df[column].dtype
        if before_type != after_type:
            print_analysis(f"Changed type from {before_type} to {after_type}", indent=4)
        else:
            print_analysis("No type conversion needed", indent=4)

        # Handle missing values
        print_analysis("Handling missing values...", indent=2)
        before_missing = processed_df[column].isna().sum()
        processed_df = preprocessor._handle_missing_values(processed_df, column, analysis)
        after_missing = processed_df[column].isna().sum()
        if before_missing != after_missing:
            print_analysis(f"Filled {before_missing - after_missing} missing values", indent=4)
            if after_missing > 0:
                print_analysis(f"Remaining missing: {after_missing}", indent=4)
        else:
            print_analysis("No missing values to handle", indent=4)

        # Handle outliers
        if pd.api.types.is_numeric_dtype(processed_df[column]):
            print_analysis("Handling outliers...", indent=2)
            before_stats = {
                'min': processed_df[column].min(),
                'max': processed_df[column].max(),
                'std': processed_df[column].std()
            }
            processed_df = preprocessor._handle_outliers(processed_df, column, analysis)
            after_stats = {
                'min': processed_df[column].min(),
                'max': processed_df[column].max(),
                'std': processed_df[column].std()
            }
            if before_stats != after_stats:
                print_analysis(
                    f"Before: min={before_stats['min']:.2f}, max={before_stats['max']:.2f}, std={before_stats['std']:.2f}",
                    indent=4)
                print_analysis(
                    f"After: min={after_stats['min']:.2f}, max={after_stats['max']:.2f}, std={after_stats['std']:.2f}",
                    indent=4)
            else:
                print_analysis("No outlier treatment applied", indent=4)

        # Apply transformations
        print_analysis("Applying transformations...", indent=2)
        before_cols = set(processed_df.columns)
        processed_df = preprocessor._apply_transformations(processed_df, column, analysis)
        after_cols = set(processed_df.columns)
        new_cols = after_cols - before_cols
        if new_cols:
            print_analysis(f"Created new columns: {', '.join(new_cols)}", indent=4)
        else:
            print_analysis("No transformations applied", indent=4)

        # Feature engineering
        print_analysis("Engineering features...", indent=2)
        before_cols = set(processed_df.columns)
        processed_df = preprocessor._engineer_features(processed_df, column, analysis)
        after_cols = set(processed_df.columns)
        new_cols = after_cols - before_cols
        if new_cols:
            print_analysis(f"Created new features: {', '.join(new_cols)}", indent=4)
        else:
            print_analysis("No feature engineering applied", indent=4)

    # Apply overall recommendations
    print_subheader("Applying overall recommendations")
    overall_recommendations = analysis_results.get('overall_recommendations', [])
    for i, rec in enumerate(overall_recommendations, 1):
        print_analysis(f"Recommendation {i}: {rec}")

    before_cols = set(processed_df.columns)
    processed_df = preprocessor._apply_overall_recommendations(processed_df)
    after_cols = set(processed_df.columns)
    new_cols = after_cols - before_cols
    if new_cols:
        print_analysis(f"Created new columns from overall recommendations: {', '.join(new_cols)}")

    # Process text features
    print_subheader("Processing text features")
    before_cols = set(processed_df.columns)
    processed_df = preprocessor._process_text_features(processed_df)
    after_cols = set(processed_df.columns)
    new_cols = after_cols - before_cols
    if new_cols:
        print_analysis(f"Created text features: {', '.join(new_cols)}")
    else:
        print_analysis("No text features created")

    # Create advanced features
    print_subheader("Creating advanced features")
    before_cols = set(processed_df.columns)
    processed_df = preprocessor._create_advanced_features(processed_df)
    after_cols = set(processed_df.columns)
    new_cols = after_cols - before_cols
    if new_cols:
        print_analysis(f"Created advanced features: {', '.join(new_cols)}")
    else:
        print_analysis("No advanced features created")

    print_header("PREPROCESSING COMPLETE")
    print_analysis(f"Original shape: {df.shape}")
    print_analysis(f"Processed shape: {processed_df.shape}")

    return processed_df


def display_dataset_summary(df, title="Dataset Summary"):
    """
    Display a summary of the dataset

    Args:
        df: DataFrame to summarize
        title: Title for the summary
    """
    print_subheader(title)
    print_analysis(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Data types summary
    dtypes_count = df.dtypes.value_counts().to_dict()
    print_analysis("Data types:")
    for dtype, count in dtypes_count.items():
        print_analysis(f"  {dtype}: {count} columns", indent=2)

    # Missing values summary
    missing = df.isna().sum()
    cols_with_missing = missing[missing > 0]
    if not cols_with_missing.empty:
        print_analysis(f"Missing values: {cols_with_missing.sum()} total in {len(cols_with_missing)} columns")
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            print_analysis(f"  {col}: {count} ({pct:.2f}%)", indent=2)
    else:
        print_analysis("No missing values detected")

    # Sample rows
    print_analysis("Sample data:")
    sample_rows = min(5, len(df))
    sample_cols = min(10, len(df.columns))
    if len(df.columns) > sample_cols:
        print(df.iloc[:sample_rows, :sample_cols].to_string())
        print(f"... and {len(df.columns) - sample_cols} more columns")
    else:
        print(df.head(sample_rows).to_string())


def display_analysis_results(analysis_results):
    """
    Display the analysis results in a readable format

    Args:
        analysis_results: Analysis results from GPT-5
    """
    print_header("GPT-5 ANALYSIS RESULTS")

    # Dataset quality score
    if 'dataset_quality_score' in analysis_results:
        quality_score = analysis_results['dataset_quality_score']
        print_analysis(f"Dataset Quality Score: {quality_score}/10")

    # Overall recommendations
    if 'overall_recommendations' in analysis_results:
        print_subheader("Overall Recommendations")
        for i, rec in enumerate(analysis_results['overall_recommendations'], 1):
            print_analysis(f"{i}. {rec}")

    # Potential issues
    if 'potential_issues' in analysis_results and analysis_results['potential_issues']:
        print_subheader("Potential Issues")
        for i, issue in enumerate(analysis_results['potential_issues'], 1):
            print_analysis(f"{i}. {issue}")

    # Time series detection
    if analysis_results.get('is_time_series', False) or 'time_series_info' in analysis_results:
        print_subheader("Time Series Characteristics")
        if 'time_series_info' in analysis_results and analysis_results['time_series_info'].get('is_time_series', False):
            print_analysis("This dataset has time series characteristics")
            for col, info in analysis_results['time_series_info'].get('time_columns', {}).items():
                if 'interval' in info:
                    print_analysis(f"Column '{col}' has regular time intervals", indent=2)
                else:
                    print_analysis(f"Column '{col}' has time data but irregular intervals", indent=2)
        else:
            print_analysis("No time series characteristics detected")

    # Special handling
    if 'special_handling' in analysis_results:
        print_subheader("Special Handling Required")
        special = analysis_results['special_handling']

        if 'text_columns' in special and special['text_columns']:
            print_analysis(f"Text columns detected: {', '.join(special['text_columns'])}")

        if 'date_columns' in special and special['date_columns']:
            print_analysis(f"Date columns detected: {', '.join(special['date_columns'])}")

        if 'target_column' in special and special['target_column']:
            print_analysis(f"Potential target column: {special['target_column']}")

    # Display sample of column analyses
    if 'column_analyses' in analysis_results:
        print_subheader("Column-specific Analysis (Sample)")
        cols = list(analysis_results['column_analyses'].keys())
        sample_size = min(5, len(cols))
        sample_cols = cols[:sample_size]

        for col in sample_cols:
            analysis = analysis_results['column_analyses'][col]
            print_analysis(f"Column: {col}")
            for key, value in analysis.items():
                print_analysis(f"{key}: {value}", indent=2)
            print()

        if len(cols) > sample_size:
            print_analysis(f"... and {len(cols) - sample_size} more columns")


def main():
    parser = argparse.ArgumentParser(description='Data Science Agent for Dataset Preprocessing')
    parser.add_argument('--input', type=str, help='Path to input dataset (CSV or Excel)')
    parser.add_argument('--output', type=str, help='Path to save processed dataset')
    parser.add_argument('--api_base', type=str, default=os.getenv('GPT5_API_BASE', "https://api.aimlapi.com/v1"),
                        help='GPT-5 API base URL')
    parser.add_argument('--api_key', type=str, default=os.getenv('GPT5_API_KEY', "361a27a60f6b4138982fd15278917fed"),
                        help='GPT-5 API key')
    parser.add_argument('--model', type=str, default="openai/gpt-5-chat-latest", help='GPT-5 model name')
    parser.add_argument('--save_analysis', type=str, help='Path to save analysis results (optional)')
    parser.add_argument('--report', type=str, help='Path to save HTML report (optional)')
    parser.add_argument('--use_gui', action='store_true', help='Use GUI for file selection')
    parser.add_argument('--verbose', action='store_true', help='Show detailed processing information')

    args = parser.parse_args()

    # Set more verbose logging if requested
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print_header("DATA SCIENCE AGENT")
    print_analysis("Initializing data science agent with GPT-5 integration")

    # Use file dialog if specified or if input/output not provided
    if args.use_gui or (not args.input or not args.output):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()

            print_analysis("Using GUI for file selection")

            # If input not provided, ask for it
            if not args.input:
                print_subheader("Select Input Dataset")
                print_analysis("Please select your input dataset file (CSV or Excel)...")
                args.input = filedialog.askopenfilename(
                    title="Select input dataset",
                    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
                )

            if not args.input:
                print_analysis("No input file selected. Exiting.", indent=2)
                return
            else:
                print_analysis(f"Selected input file: {args.input}", indent=2)

            # If output not provided, ask for it
            if not args.output:
                print_subheader("Select Output Location")
                print_analysis("Please select where to save the processed dataset...")
                args.output = filedialog.asksaveasfilename(
                    title="Save processed dataset as",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
                )

            if not args.output:
                print_analysis("No output location selected. Exiting.", indent=2)
                return
            else:
                print_analysis(f"Selected output location: {args.output}", indent=2)

            # Ask about HTML report
            if not args.report:
                report_prompt = input("\nDo you want to generate an HTML report? (yes/no): ").lower()
                if report_prompt.startswith('y'):
                    args.report = filedialog.asksaveasfilename(
                        title="Save HTML report as",
                        defaultextension=".html",
                        filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
                    )
                    if args.report:
                        print_analysis(f"HTML report will be saved to: {args.report}", indent=2)

            # Ask about saving analysis
            if not args.save_analysis:
                analysis_prompt = input("\nDo you want to save the analysis results? (yes/no): ").lower()
                if analysis_prompt.startswith('y'):
                    args.save_analysis = filedialog.asksaveasfilename(
                        title="Save analysis results as",
                        defaultextension=".json",
                        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                    )
                    if args.save_analysis:
                        print_analysis(f"Analysis results will be saved to: {args.save_analysis}", indent=2)

        except ImportError:
            print_analysis("GUI libraries not available. Please provide input/output paths as arguments.")
            return

    # Hardcode paths if not provided and not using GUI
    if not args.input:
        args.input = "dataset.csv"  # Default input path
        print_analysis(f"Using default input path: {args.input}")

    if not args.output:
        args.output = "processed_dataset.csv"  # Default output path
        print_analysis(f"Using default output path: {args.output}")

    try:
        # Step 1: Load the dataset
        print_step(1, 4, "Loading and examining dataset")
        print_analysis(f"Loading dataset from {args.input}")
        data = load_dataset(args.input)
        df = data['dataframe']
        metadata = data['metadata']

        print_analysis(f"Dataset loaded successfully: {metadata['rows']} rows, {metadata['columns']} columns", indent=2)

        # Display dataset summary
        display_dataset_summary(df)

        # Step 2: Analyze the dataset with GPT-5
        print_step(2, 4, "Analyzing dataset with GPT-5")
        print_thinking("GPT-5 is analyzing the dataset structure, patterns, and characteristics...")

        analyzer = DataAnalyzer(api_base=args.api_base, api_key=args.api_key, model=args.model)
        analysis_results = analyzer.analyze_dataset(df)

        print_analysis("GPT-5 analysis complete", indent=2)

        # Display analysis results
        display_analysis_results(analysis_results)

        # Save analysis results if requested
        if args.save_analysis:
            print_analysis(f"Saving analysis results to {args.save_analysis}")
            with open(args.save_analysis, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print_analysis("Analysis results saved successfully", indent=2)

        # Step 3: Preprocess the dataset based on the analysis
        print_step(3, 4, "Preprocessing dataset based on analysis")
        if len(df) * len(df.columns) > 10000:  # Use progress tracking for larger datasets
            print_analysis("Large dataset detected. Using detailed progress tracking.")
            processed_df = process_with_progress(df, analysis_results)
        else:
            print_analysis("Processing dataset with detailed steps:")
            preprocessor = DataPreprocessor(analysis_results)
            processed_df = preprocessor.preprocess_dataset(df)

        # Display summary of processed dataset
        display_dataset_summary(processed_df, "Processed Dataset Summary")

        # Step 4: Save the processed dataset
        print_step(4, 4, "Saving processed dataset and generating reports")

        file_ext = os.path.splitext(args.output)[1].lower()
        print_analysis(f"Saving processed dataset to {args.output}")

        if file_ext == '.csv':
            processed_df.to_csv(args.output, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            processed_df.to_excel(args.output, index=False)
        else:
            processed_df.to_csv(args.output + '.csv', index=False)
            print_analysis(f"Unknown output format '{file_ext}'. Saved as CSV instead.", indent=2)

        print_analysis("Processed dataset saved successfully", indent=2)

        # Compare original and processed datasets
        print_subheader("Dataset Transformation Summary")
        print_analysis(f"Original shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print_analysis(f"Processed shape: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns")

        # Show column changes
        original_cols = set(df.columns)
        processed_cols = set(processed_df.columns)
        new_cols = processed_cols - original_cols
        if new_cols:
            print_analysis(f"New columns created: {len(new_cols)}")
            print_analysis(f"  {', '.join(list(new_cols)[:10])}" +
                           (f" and {len(new_cols) - 10} more..." if len(new_cols) > 10 else ""), indent=2)

        removed_cols = original_cols - processed_cols
        if removed_cols:
            print_analysis(f"Columns removed: {len(removed_cols)}")
            print_analysis(f"  {', '.join(removed_cols)}", indent=2)

        # Generate HTML report if requested
        if args.report:
            print_analysis("Generating HTML report with visualizations")
            generate_html_report(df, processed_df, analysis_results, args.report)

        print_header("PROCESSING COMPLETE")
        print_analysis(f"Original dataset: {args.input}")
        print_analysis(f"Processed dataset: {args.output}")
        if args.report:
            print_analysis(f"HTML report: {args.report}")

    except Exception as e:
        print_header("ERROR OCCURRED")
        print_analysis(f"Error in data science agent: {str(e)}")
        import traceback
        print_analysis("Detailed error information:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()