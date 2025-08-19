import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> Dict[str, Any]:
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
        df = standardize_missing_values(df)
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
    missing_indicators = ["?", "NA", "N/A", "NULL", "None", "unknown", "missing", "-", ""]
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].replace(missing_indicators, np.nan)

            if df[column].dtype == 'object':
                for indicator in missing_indicators:
                    mask = df[column].str.lower() == indicator.lower()
                    df.loc[mask, column] = np.nan

    return df


def smart_sample(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    target_col = None
    for col in df.columns:
        if col.lower() in ['target', 'label', 'class', 'y', 'outcome', 'result']:
            target_col = col
            break

    if target_col and df[target_col].nunique() < 10:
        from sklearn.model_selection import train_test_split
        _, sample_df = train_test_split(
            df,
            test_size=max_rows / len(df),
            stratify=df[target_col],
            random_state=42
        )
        return sample_df
    indices = np.linspace(0, len(df) - 1, max_rows).astype(int)
    return df.iloc[indices]