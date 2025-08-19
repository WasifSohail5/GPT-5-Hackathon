import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import multiprocessing
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.transformers = {}  # Store fitted transformers for potential reuse

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        processed_df = self._handle_custom_missing_values(processed_df)
        use_parallel = len(df) * len(df.columns) > 100000
        if use_parallel:
            return self.preprocess_dataset_parallel(df)
        else:
            column_analyses = self.analysis_results.get('column_analyses', {})

            for column, analysis in column_analyses.items():
                if column not in processed_df.columns:
                    logger.warning(f"Column {column} not found in DataFrame. Skipping.")
                    continue
                logger.info(f"Processing column: {column}")
                processed_df = self._convert_data_types(processed_df, column, analysis)
                processed_df = self._handle_missing_values(processed_df, column, analysis)
                processed_df = self._handle_outliers(processed_df, column, analysis)
                processed_df = self._apply_transformations(processed_df, column, analysis)
                processed_df = self._engineer_features(processed_df, column, analysis)
            processed_df = self._apply_overall_recommendations(processed_df)
            processed_df = self._process_text_features(processed_df)
            processed_df = self._create_advanced_features(processed_df)
            return processed_df

    def preprocess_dataset_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        processed_df = self._handle_custom_missing_values(processed_df)
        column_analyses = self.analysis_results.get('column_analyses', {})
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using parallel processing with {num_cores} cores")
        processed_series = Parallel(n_jobs=num_cores)(
            delayed(self._parallel_process_column)(processed_df, column, analysis)
            for column, analysis in column_analyses.items()
            if column in processed_df.columns
        )

        for i, (column, _) in enumerate([(c, a) for c, a in column_analyses.items() if c in df.columns]):
            processed_df[column] = processed_series[i]
        processed_df = self._apply_overall_recommendations(processed_df)
        processed_df = self._process_text_features(processed_df)
        processed_df = self._create_advanced_features(processed_df)

        return processed_df

    def _parallel_process_column(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.Series:
        series = df[column].copy()
        missing_strategy = analysis.get('missing_values', '').lower()
        if pd.isna(series).any():
            series = self._handle_missing_values_series(series, missing_strategy)
        outlier_strategy = analysis.get('outliers', '').lower()
        if pd.api.types.is_numeric_dtype(series) and not ('none' in outlier_strategy or 'keep' in outlier_strategy):
            series = self._handle_outliers_series(series, outlier_strategy)
        transformations = analysis.get('transformations', '').lower()
        if pd.api.types.is_numeric_dtype(series) and transformations:
            if 'log' in transformations:
                min_val = series.min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    series = np.log(series + offset)
                else:
                    series = np.log(series)

            if 'standard' in transformations or 'z-score' in transformations:
                scaler = StandardScaler()
                reshaped = series.values.reshape(-1, 1)
                series = pd.Series(scaler.fit_transform(reshaped).ravel(), index=series.index)

            if 'minmax' in transformations or 'normalize' in transformations:
                scaler = MinMaxScaler()
                reshaped = series.values.reshape(-1, 1)
                series = pd.Series(scaler.fit_transform(reshaped).ravel(), index=series.index)

        return series

    def _handle_custom_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_indicators = ["?", "NA", "N/A", "NULL", "None", "unknown", "missing", "-", ""]
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].replace(missing_indicators, np.nan)
                if df[column].dtype == 'object':
                    for indicator in missing_indicators:
                        mask = df[column].str.lower() == indicator.lower()
                        df.loc[mask, column] = np.nan

        return df

    def _convert_data_types(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        data_type = analysis.get('data_type', '').lower()

        try:
            if 'int' in data_type or 'float' in data_type:
                if 'int' in data_type:
                    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                else:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
            elif 'date' in data_type or 'time' in data_type:
                df[column] = pd.to_datetime(df[column], errors='coerce')
            elif 'categorical' in data_type:
                df[column] = df[column].astype('category')
            elif 'bool' in data_type:
                df[column] = df[column].astype(bool)
            elif 'string' in data_type or 'text' in data_type:
                df[column] = df[column].astype(str)
        except Exception as e:
            logger.error(f"Error converting data type for column {column}: {str(e)}")

        return df

    def _handle_missing_values(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        strategy = analysis.get('missing_values', '').lower()

        if not pd.isna(df[column]).any():
            return df

        try:
            df[column] = self._handle_missing_values_series(df[column], strategy)
        except Exception as e:
            logger.error(f"Error handling missing values for column {column}: {str(e)}")

        return df

    def _handle_missing_values_series(self, series: pd.Series, strategy: str) -> pd.Series:
        if 'drop' in strategy:
            return series
        elif 'mean' in strategy:
            return series.fillna(series.mean())
        elif 'median' in strategy:
            return series.fillna(series.median())
        elif 'mode' in strategy:
            return series.fillna(series.mode()[0] if not series.mode().empty else np.nan)
        elif 'constant' in strategy or 'value' in strategy:
            import re
            match = re.search(r'value[:|=|\s]+([0-9.-]+)', strategy)
            fill_value = float(match.group(1)) if match else 0
            return series.fillna(fill_value)
        elif 'knn' in strategy:
            if pd.api.types.is_numeric_dtype(series):
                imputer = KNNImputer(n_neighbors=5)
                reshaped = series.values.reshape(-1, 1)
                return pd.Series(imputer.fit_transform(reshaped).ravel(), index=series.index)
            else:
                return series.fillna(series.mode()[0] if not series.mode().empty else np.nan)
        else:
            return series

    def _handle_outliers(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        outlier_strategy = analysis.get('outliers', '').lower()

        if not pd.api.types.is_numeric_dtype(df[column]) or 'none' in outlier_strategy or 'keep' in outlier_strategy:
            return df

        try:
            df[column] = self._handle_outliers_series(df[column], outlier_strategy)
        except Exception as e:
            logger.error(f"Error handling outliers for column {column}: {str(e)}")

        return df

    def _handle_outliers_series(self, series: pd.Series, outlier_strategy: str) -> pd.Series:
        if 'isolation' in outlier_strategy or 'forest' in outlier_strategy:
            mask_valid = ~np.isnan(series.values)
            if mask_valid.sum() > 10:
                X = series.values[mask_valid].reshape(-1, 1)
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                iso_forest.fit(X)
                is_outlier = iso_forest.predict(X) == -1

                if 'cap' in outlier_strategy:
                    non_outliers = X[~is_outlier].flatten()
                    if len(non_outliers) > 0:
                        lower = np.percentile(non_outliers, 5)
                        upper = np.percentile(non_outliers, 95)
                        result = series.copy()
                        outlier_indices = np.where(mask_valid)[0][is_outlier]
                        result.iloc[outlier_indices] = result.iloc[outlier_indices].clip(lower, upper)
                        return result
                elif 'remove' in outlier_strategy:
                    result = series.copy()
                    outlier_indices = np.where(mask_valid)[0][is_outlier]
                    result.iloc[outlier_indices] = np.nan
                    return result

        if 'iqr' in outlier_strategy:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if 'cap' in outlier_strategy:
                return series.clip(lower=lower_bound, upper=upper_bound)
            elif 'remove' in outlier_strategy:
                result = series.copy()
                mask = (series < lower_bound) | (series > upper_bound)
                result[mask] = np.nan
                return result

        elif 'zscore' in outlier_strategy:
            threshold = 3.0
            import re
            match = re.search(r'threshold[:|=|\s]+([0-9.]+)', outlier_strategy)
            if match:
                threshold = float(match.group(1))

            z_scores = np.abs((series - series.mean()) / series.std())

            if 'cap' in outlier_strategy:
                mean_val = series.mean()
                std_val = series.std()
                mask = z_scores > threshold
                result = series.copy()
                result[mask] = np.sign(series[mask] - mean_val) * threshold * std_val + mean_val
                return result
            elif 'remove' in outlier_strategy:
                result = series.copy()
                mask = z_scores > threshold
                result[mask] = np.nan
                return result
        return series

    def _detect_anomalies(self, df: pd.DataFrame, column: str) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return pd.Series(False, index=df.index)

        X = df[column].values.reshape(-1, 1)

        mask_valid = ~np.isnan(X).ravel()
        X_valid = X[mask_valid]

        if len(X_valid) < 10:  # Not enough data
            return pd.Series(False, index=df.index)

        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_forest.fit(X_valid)
        result = pd.Series(False, index=df.index)
        result.iloc[mask_valid] = iso_forest.predict(X_valid) == -1

        return result

    def _apply_transformations(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        transformations = analysis.get('transformations', '').lower()

        if not pd.api.types.is_numeric_dtype(df[column]) or not transformations:
            return df

        try:
            if 'log' in transformations:
                if (df[column] <= 0).any():
                    min_val = df[column].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        df[column + '_log'] = np.log(df[column] + offset)
                    else:
                        df[column + '_log'] = np.log(df[column])
                else:
                    df[column + '_log'] = np.log(df[column])

            if 'standard' in transformations or 'z-score' in transformations:
                if column not in self.transformers:
                    scaler = StandardScaler()
                    self.transformers[column] = {'scaler': scaler}
                    reshaped = df[column].values.reshape(-1, 1)
                    df[column + '_scaled'] = scaler.fit_transform(reshaped).ravel()
                else:
                    scaler = self.transformers[column]['scaler']
                    reshaped = df[column].values.reshape(-1, 1)
                    df[column + '_scaled'] = scaler.transform(reshaped).ravel()

            if 'minmax' in transformations or 'normalize' in transformations:
                if column not in self.transformers:
                    scaler = MinMaxScaler()
                    self.transformers[column] = {'minmax_scaler': scaler}
                    reshaped = df[column].values.reshape(-1, 1)
                    df[column + '_norm'] = scaler.fit_transform(reshaped).ravel()
                else:
                    scaler = self.transformers[column]['minmax_scaler']
                    reshaped = df[column].values.reshape(-1, 1)
                    df[column + '_norm'] = scaler.transform(reshaped).ravel()

            if 'onehot' in transformations or 'one-hot' in transformations:
                unique_values = df[column].nunique()
                if unique_values < 15:  # Only one-hot encode if there aren't too many categories
                    dummies = pd.get_dummies(df[column], prefix=column)
                    df = pd.concat([df, dummies], axis=1)

            if 'label' in transformations:
                if column not in self.transformers:
                    encoder = LabelEncoder()
                    self.transformers[column] = {'label_encoder': encoder}
                    df[column + '_encoded'] = encoder.fit_transform(df[column].astype(str))
                else:
                    encoder = self.transformers[column]['label_encoder']
                    # Handle unseen labels
                    df[column + '_encoded'] = df[column].apply(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1
                    )

        except Exception as e:
            logger.error(f"Error applying transformations for column {column}: {str(e)}")

        return df

    def _engineer_features(self, df: pd.DataFrame, column: str, analysis: Dict[str, Any]) -> pd.DataFrame:
        feature_engineering = analysis.get('feature_engineering', '')

        if not feature_engineering:
            return df

        try:
            # Handle date/time features
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                df[f"{column}_year"] = df[column].dt.year
                df[f"{column}_month"] = df[column].dt.month
                df[f"{column}_day"] = df[column].dt.day
                df[f"{column}_dayofweek"] = df[column].dt.dayofweek
                df[f"{column}_quarter"] = df[column].dt.quarter
                df[f"{column}_is_weekend"] = df[column].dt.dayofweek > 4

            # Create polynomial features if suggested
            if 'polynomial' in feature_engineering.lower() and pd.api.types.is_numeric_dtype(df[column]):
                df[f"{column}_squared"] = df[column] ** 2
                df[f"{column}_cubed"] = df[column] ** 3

            # Create binned features if suggested
            if 'bin' in feature_engineering.lower() and pd.api.types.is_numeric_dtype(df[column]):
                df[f"{column}_binned"] = pd.qcut(df[column], q=5, duplicates='drop', labels=False)

        except Exception as e:
            logger.error(f"Error engineering features for column {column}: {str(e)}")

        return df

    def _apply_overall_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        recommendations = self.analysis_results.get('overall_recommendations', [])

        for rec in recommendations:
            rec_lower = rec.lower()
            if 'pca' in rec_lower or 'dimensionality reduction' in rec_lower:
                try:
                    # Select only numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 2:
                        components = min(len(numeric_cols) - 1, 10)
                        # Extract number of components if specified
                        import re
                        match = re.search(r'components[:|=|\s]+([0-9]+)', rec_lower)
                        if match:
                            components = int(match.group(1))

                        pca = PCA(n_components=components)
                        pca_result = pca.fit_transform(df[numeric_cols].fillna(0))

                        # Add PCA columns
                        for i in range(components):
                            df[f'pca_component_{i + 1}'] = pca_result[:, i]

                        logger.info(f"Applied PCA with {components} components")

                        # Log explained variance
                        explained_var = np.sum(pca.explained_variance_ratio_)
                        logger.info(f"Total explained variance: {explained_var:.2%}")
                except Exception as e:
                    logger.error(f"Error applying PCA: {str(e)}")

        return df

    def _process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        text_columns = self.analysis_results.get('special_handling', {}).get('text_columns', [])
        if not text_columns:
            text_columns = []
            for col in df.select_dtypes(include=['object']).columns:
                # Check average string length
                avg_length = df[col].astype(str).str.len().mean()
                num_words = df[col].astype(str).str.split().str.len().mean()

                if avg_length > 50 or num_words > 10:  # Heuristic for text data
                    text_columns.append(col)

        # Process text columns
        for col in text_columns:
            try:
                # Basic text cleaning
                df[f"{col}_cleaned"] = df[col].astype(str).str.lower()
                df[f"{col}_cleaned"] = df[f"{col}_cleaned"].str.replace(r'[^\w\s]', '', regex=True)

                # Extract simple features
                df[f"{col}_char_count"] = df[col].astype(str).str.len()
                df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()

                # TF-IDF for top features (if not too many rows)
                if len(df) < 10000:
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        tfidf = TfidfVectorizer(max_features=5, stop_words='english')
                        tfidf_matrix = tfidf.fit_transform(df[col].fillna('').astype(str))

                        # Get feature names
                        feature_names = tfidf.get_feature_names_out()

                        # Add TF-IDF features
                        for i, feature in enumerate(feature_names):
                            df[f"{col}_tfidf_{feature}"] = tfidf_matrix.getcol(i).toarray()
                    except Exception as e:
                        logger.warning(f"Error creating TF-IDF features for {col}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing text column {col}: {str(e)}")

        return df

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) > 1:
                # Create ratios between important numeric columns (limit to avoid explosion)
                important_cols = numeric_cols[:5]
                for i in range(len(important_cols)):
                    for j in range(i + 1, len(important_cols)):
                        col1 = important_cols[i]
                        col2 = important_cols[j]

                        # Create ratio if it makes sense (no zeros in denominator)
                        if (df[col2] != 0).all():
                            ratio_name = f"{col1}_to_{col2}_ratio"
                            df[ratio_name] = df[col1] / df[col2]

            # Get datetime columns
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)

            # Extract datetime features if not already done
            for col in datetime_cols:
                if f"{col}_year" not in df.columns:
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    df[f"{col}_quarter"] = df[col].dt.quarter
                    df[f"{col}_is_weekend"] = df[col].dt.dayofweek > 4

        except Exception as e:
            logger.error(f"Error creating advanced features: {str(e)}")

        return df