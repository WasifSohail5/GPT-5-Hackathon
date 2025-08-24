import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging
import openpyxl
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, api_base: str, api_key: str, model: str = "openai/gpt-5-chat-latest"):
        self.chat = ChatOpenAI(
            openai_api_base=api_base,
            openai_api_key=api_key,
            model=model
        )

    def _get_data_summary(self, df: pd.DataFrame) -> str:
        summary = []
        summary.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = df[col].isna().sum()
            missing_pct = round((missing / len(df)) * 100, 2)

            if pd.api.types.is_numeric_dtype(df[col]):
                summary.append(f"Column '{col}' (type: {col_type}):")
                summary.append(f"  - Range: {df[col].min()} to {df[col].max()}")
                summary.append(f"  - Mean: {df[col].mean():.2f}, Median: {df[col].median()}")
                summary.append(f"  - Missing values: {missing} ({missing_pct}%)")
            else:
                unique_vals = df[col].nunique()
                summary.append(f"Column '{col}' (type: {col_type}):")
                summary.append(f"  - Unique values: {unique_vals}")
                summary.append(f"  - Missing values: {missing} ({missing_pct}%)")
                if unique_vals <= 10:
                    summary.append(f"  - Value counts: {dict(df[col].value_counts().head(5))}")
                if col_type == 'object' and unique_vals > 10:
                    examples = df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()
                    summary.append(f"  - Examples: {examples}")
        missing_patterns = self._detect_missing_patterns(df)
        if missing_patterns:
            summary.append("\nMissing Value Patterns:")
            if 'empty_rows' in missing_patterns:
                summary.append(f"  - Empty rows: {missing_patterns['empty_rows']['count']} " +
                               f"({missing_patterns['empty_rows']['percentage']}%)")

            if 'correlated_missing' in missing_patterns:
                summary.append("  - Correlated missing values between:")
                for col1, col2, corr in missing_patterns['correlated_missing']['pairs'][:3]:  # Show top 3
                    summary.append(f"    * {col1} and {col2} (correlation: {corr:.2f})")

        time_series_info = self._detect_time_series(df)
        if time_series_info.get('is_time_series', False):
            summary.append("\nTime Series Characteristics Detected:")
            for col, info in time_series_info.get('time_columns', {}).items():
                if 'interval' in info:
                    summary.append(f"  - Column '{col}' has regular time intervals: {info['interval']}")
                else:
                    summary.append(f"  - Column '{col}' has time data but irregular intervals")

        text_columns = self._detect_text_columns(df)
        if text_columns:
            summary.append("\nText Data Detected:")
            for col, info in text_columns.items():
                summary.append(f"  - Column '{col}': avg length {info['avg_length']:.1f} chars, " +
                               f"{info['avg_words']:.1f} words")

        return "\n".join(summary)

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) > 10000:
            from data_loader import smart_sample
            logger.info(f"Dataset is large ({len(df)} rows). Creating a representative sample for analysis.")
            analysis_df = smart_sample(df)
        else:
            analysis_df = df
        data_summary = self._get_data_summary(analysis_df)
        system_prompt = """
        You are a data science expert. Analyze the dataset summary provided and give detailed recommendations for preprocessing.
        Focus on:
        1. Data types and necessary conversions
        2. Missing values and appropriate handling strategies
        3. Outlier detection and handling
        4. Feature engineering opportunities
        5. Scaling/normalization needs
        6. Encoding categorical variables
        7. Dimensionality reduction if needed
        8. Text data processing if present
        9. Time series handling if applicable

        Provide your analysis in a structured JSON format with the following keys:
        {
            "column_analyses": {
                "column_name": {
                    "data_type": "suggested data type",
                    "missing_values": "strategy and justification",
                    "outliers": "detection method and handling strategy",
                    "transformations": "suggested transformations (scaling, normalization, etc.)",
                    "feature_engineering": "suggestions for derived features if applicable"
                }
                // repeat for each column
            },
            "overall_recommendations": [
                // list of preprocessing steps in recommended order
            ],
            "potential_issues": [
                // list of potential issues to be aware of
            ],
            "dataset_quality_score": 0-10, // overall quality score
            "is_time_series": true/false, // whether dataset has time series characteristics
            "special_handling": { // any special handling recommendations
                "text_columns": ["col1", "col2"], // columns with text data
                "date_columns": ["col3"], // date/time columns
                "target_column": "col4" // potential target column if detected
            }
        }

        Ensure your recommendations are specific to the data types and patterns observed.
        """

        human_message = f"Here's a summary of the dataset I need to preprocess for machine learning:\n\n{data_summary}"

        logger.info("Sending dataset summary to GPT-5 for analysis")
        response = self.chat.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])
        try:
            analysis_results = json.loads(response.content)
            logger.info("Successfully parsed GPT-5 analysis response")

            analysis_results['missing_patterns'] = self._detect_missing_patterns(df)
            analysis_results['time_series_info'] = self._detect_time_series(df)
            analysis_results['text_columns_info'] = self._detect_text_columns(df)

            return analysis_results
        except json.JSONDecodeError:
            logger.error("Failed to parse GPT-5 response as JSON")
            return {
                "error": "Failed to parse GPT-5 response as JSON",
                "raw_response": response.content
            }

    def _detect_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_rows = len(df)
        patterns = {}
        empty_rows = df.isna().all(axis=1).sum()
        if empty_rows > 0:
            patterns['empty_rows'] = {
                'count': int(empty_rows),
                'percentage': round((empty_rows / total_rows) * 100, 2),
                'recommendation': 'Consider removing completely empty rows'
            }
        missing_matrix = df.isna().astype(int)
        corr_matrix = missing_matrix.corr()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            patterns['correlated_missing'] = {
                'pairs': high_corr_pairs,
                'recommendation': 'Missing values in these column pairs are correlated, consider joint imputation'
            }

        return patterns

    def _detect_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        date_columns = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue

        if not date_columns:
            return {'is_time_series': False}
        time_series_candidates = {}

        for col in date_columns:
            dates = pd.to_datetime(df[col])
            if dates.is_monotonic_increasing:
                intervals = dates.diff().dropna()
                if len(intervals) > 0:
                    most_common_interval = intervals.mode()[0]
                    consistency = (abs(intervals - most_common_interval) < pd.Timedelta('1 minute')).mean()
                    if consistency > 0.8:
                        time_series_candidates[col] = {
                            'interval': most_common_interval,
                            'consistency': consistency,
                            'is_sorted': True
                        }
                    else:
                        time_series_candidates[col] = {
                            'is_sorted': True,
                            'irregular_interval': True
                        }

        if time_series_candidates:
            return {
                'is_time_series': True,
                'time_columns': time_series_candidates,
                'recommendations': [
                    'Extract features from dates (day of week, month, etc.)',
                    'Consider lag features for prediction tasks',
                    'Check for seasonality in numeric variables'
                ]
            }

        return {'is_time_series': False}

    def _detect_text_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        text_columns = {}
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isna().mean() > 0.5:
                continue
            avg_length = df[col].astype(str).str.len().mean()
            avg_words = df[col].astype(str).str.split().str.len().mean()

            if avg_length > 50 or avg_words > 7:
                text_columns[col] = {
                    'avg_length': avg_length,
                    'avg_words': avg_words,
                    'recommendation': 'Consider text preprocessing and feature extraction'
                }
        return text_columns