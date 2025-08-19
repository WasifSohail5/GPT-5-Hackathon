import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(file_path: str) -> Dict[str, Any]:
    """
    Load dataset from CSV or Excel file

    Args:
        file_path: Path to the dataset file

    Returns:
        Dict containing dataframe and metadata
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
            file_type = 'CSV'
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            file_type = 'Excel'
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please provide CSV or Excel file.")

        # Standardize missing values
        df = standardize_missing_values(df)

        # Extract basic metadata
        metadata = {
            'file_name': os.path.basename(file_path),
            'file_type': file_type,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_counts': {col: int(df[col].isna().sum()) for col in df.columns},
            'missing_percentage': {col: float(round((df[col].isna().sum() / len(df)) * 100, 2)) for col in df.columns}
        }

        return {
            'dataframe': df,
            'metadata': metadata
        }

    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and standardize custom missing value indicators

    Args:
        df: Original pandas DataFrame

    Returns:
        DataFrame with standardized missing values
    """
    # Common missing value indicators
    missing_indicators = ["?", "NA", "N/A", "NULL", "None", "unknown", "missing", "-", ""]

    # Check all columns
    for column in df.columns:
        # For string/object columns
        if df[column].dtype == 'object':
            # Replace all indicators with NaN
            df[column] = df[column].replace(missing_indicators, np.nan)

            # Case insensitive replacement
            if df[column].dtype == 'object':
                for indicator in missing_indicators:
                    mask = df[column].str.lower() == indicator.lower()
                    df.loc[mask, column] = np.nan

    return df


def smart_sample(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """
    Create a representative sample for large datasets

    Args:
        df: Original pandas DataFrame
        max_rows: Maximum number of rows in the sample

    Returns:
        Sampled DataFrame
    """
    if len(df) <= max_rows:
        return df

    # Stratified sampling for categorical target if present
    target_col = None
    # This is a heuristic to identify potential target columns
    for col in df.columns:
        if col.lower() in ['target', 'label', 'class', 'y', 'outcome', 'result']:
            target_col = col
            break

    if target_col and df[target_col].nunique() < 10:
        from sklearn.model_selection import train_test_split
        # Stratified sampling
        _, sample_df = train_test_split(
            df,
            test_size=max_rows / len(df),
            stratify=df[target_col],
            random_state=42
        )
        return sample_df

    # If no target or too many classes, use random sampling with systematic approach
    indices = np.linspace(0, len(df) - 1, max_rows).astype(int)
    return df.iloc[indices]