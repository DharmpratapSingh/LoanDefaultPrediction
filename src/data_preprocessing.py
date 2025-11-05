"""Data preprocessing module for loan default prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks."""

    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration.

        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.scaler = MinMaxScaler()
        self.pca = None
        self.smote = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, header=0)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values imputed
        """
        try:
            logger.info("Handling missing values")
            df = df.copy()

            # Log missing values before imputation
            missing_before = df.isnull().sum()
            logger.info(f"Missing values before imputation:\n{missing_before[missing_before > 0]}")

            # Impute Emp_length with mode (fix deprecated warning)
            if 'Emp_length' in df.columns and df['Emp_length'].isnull().any():
                mode_value = df['Emp_length'].mode()[0]
                df['Emp_length'] = df['Emp_length'].fillna(mode_value)
                logger.info(f"Imputed Emp_length with mode: {mode_value}")

            # Impute Rate with median (fix deprecated warning)
            if 'Rate' in df.columns and df['Rate'].isnull().any():
                median_value = df['Rate'].median()
                df['Rate'] = df['Rate'].fillna(median_value)
                logger.info(f"Imputed Rate with median: {median_value}")

            # Verify no missing values remain
            missing_after = df.isnull().sum().sum()
            logger.info(f"Total missing values after imputation: {missing_after}")

            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        try:
            logger.info("Handling outliers")
            df = df.copy()

            # Cap age at configured threshold
            age_cap = self.config['preprocessing']['age_cap']
            if 'Age' in df.columns:
                outliers_count = (df['Age'] > age_cap).sum()
                df['Age'] = df['Age'].apply(lambda x: min(x, age_cap))
                logger.info(f"Capped {outliers_count} age values at {age_cap}")

            return df
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def encode_target(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Encode the target variable to binary (0/1).

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            DataFrame with encoded target
        """
        try:
            logger.info(f"Encoding target column: {target_column}")
            df = df.copy()

            if target_column in df.columns:
                original_dist = df[target_column].value_counts()
                logger.info(f"Original target distribution:\n{original_dist}")

                df[target_column] = df[target_column].apply(lambda x: 1 if x == 'Y' else 0)

                new_dist = df[target_column].value_counts()
                logger.info(f"Encoded target distribution:\n{new_dist}")

            return df
        except Exception as e:
            logger.error(f"Error encoding target: {str(e)}")
            raise

    def encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: list,
        drop_first: bool = False
    ) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            drop_first: Whether to drop the first category

        Returns:
            DataFrame with encoded categorical variables
        """
        try:
            logger.info(f"Encoding categorical columns: {categorical_cols}")
            df = df.copy()

            df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

            logger.info(f"Columns after encoding: {df.shape[1]}")
            return df
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}")
            raise

    def scale_features(
        self,
        X: pd.DataFrame,
        numerical_cols: list,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply Min-Max scaling to numerical features.

        Args:
            X: Input features
            numerical_cols: List of numerical column names
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            Scaled features
        """
        try:
            X = X.copy()

            if fit:
                logger.info(f"Fitting scaler on {len(numerical_cols)} numerical columns")
                X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            else:
                logger.info(f"Transforming {len(numerical_cols)} numerical columns")
                X[numerical_cols] = self.scaler.transform(X[numerical_cols])

            return X
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for handling class imbalance.

        Args:
            X: Features
            y: Target variable
            random_state: Random state for reproducibility

        Returns:
            Resampled features and target
        """
        try:
            logger.info("Applying SMOTE for class balancing")

            # Log class distribution before SMOTE
            from collections import Counter
            original_dist = Counter(y)
            logger.info(f"Class distribution before SMOTE: {dict(original_dist)}")

            self.smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = self.smote.fit_resample(X, y)

            # Log class distribution after SMOTE
            new_dist = Counter(y_resampled)
            logger.info(f"Class distribution after SMOTE: {dict(new_dist)}")

            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE: {str(e)}")
            raise

    def apply_pca(
        self,
        X: np.ndarray,
        n_components: int,
        fit: bool = True
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.

        Args:
            X: Input features
            n_components: Number of components to retain
            fit: Whether to fit PCA (True for training, False for inference)

        Returns:
            Transformed features
        """
        try:
            if fit:
                logger.info(f"Applying PCA with {n_components} components")
                self.pca = PCA(n_components=n_components)
                X_pca = self.pca.fit_transform(X)

                explained_var = self.pca.explained_variance_ratio_.sum()
                logger.info(f"PCA explained variance: {explained_var:.4f}")
            else:
                if self.pca is None:
                    raise ValueError("PCA not fitted. Call with fit=True first.")
                logger.info(f"Transforming data with existing PCA")
                X_pca = self.pca.transform(X)

            logger.info(f"Shape after PCA: {X_pca.shape}")
            return X_pca
        except Exception as e:
            logger.error(f"Error applying PCA: {str(e)}")
            raise

    def preprocess_pipeline(
        self,
        file_path: str,
        apply_pca_flag: bool = True,
        for_linear_model: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.

        Args:
            file_path: Path to the raw data CSV
            apply_pca_flag: Whether to apply PCA
            for_linear_model: Whether preprocessing is for linear models

        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Starting preprocessing pipeline")

            # Load data
            df = self.load_data(file_path)

            # Handle missing values
            df = self.handle_missing_values(df)

            # Handle outliers
            df = self.handle_outliers(df)

            # Encode target variable
            target_col = self.config['preprocessing']['target_column']
            df = self.encode_target(df, target_col)

            # One-hot encode categorical variables
            cat_cols = self.config['preprocessing']['categorical_columns']
            df = self.encode_categorical(df, cat_cols, drop_first=False)

            # Drop columns for linear models if needed
            if for_linear_model:
                cols_to_drop = self.config['preprocessing']['columns_to_drop_for_linear']
                existing_cols = [col for col in cols_to_drop if col in df.columns]
                if existing_cols:
                    df = df.drop(columns=existing_cols)
                    logger.info(f"Dropped columns for linear model: {existing_cols}")

            # Separate features and target
            id_col = self.config['preprocessing']['id_column']
            X = df.drop([id_col, target_col], axis=1)
            y = df[target_col]

            # Scale numerical features
            num_cols = [col for col in self.config['preprocessing']['numerical_columns'] if col in X.columns]
            X = self.scale_features(X, num_cols, fit=True)

            # Apply SMOTE
            random_state = self.config['preprocessing']['smote_random_state']
            X_resampled, y_resampled = self.apply_smote(X, y, random_state)

            # Apply PCA if requested
            if apply_pca_flag:
                n_components = self.config['pca']['n_components']
                X_resampled = self.apply_pca(X_resampled, n_components, fit=True)

            # Train-test split
            test_size = self.config['preprocessing']['test_size']
            random_state = self.config['preprocessing']['random_state']

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled,
                test_size=test_size,
                random_state=random_state
            )

            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            logger.info("Preprocessing pipeline completed successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def save_preprocessors(self, save_dir: str):
        """
        Save fitted preprocessors to disk.

        Args:
            save_dir: Directory to save the preprocessors
        """
        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Save scaler
            scaler_path = Path(save_dir) / self.config['persistence']['scaler_filename']
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

            # Save PCA if fitted
            if self.pca is not None:
                pca_path = Path(save_dir) / self.config['persistence']['pca_filename']
                joblib.dump(self.pca, pca_path)
                logger.info(f"PCA saved to {pca_path}")

        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise

    def load_preprocessors(self, load_dir: str):
        """
        Load fitted preprocessors from disk.

        Args:
            load_dir: Directory containing the preprocessors
        """
        try:
            # Load scaler
            scaler_path = Path(load_dir) / self.config['persistence']['scaler_filename']
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")

            # Load PCA if exists
            pca_path = Path(load_dir) / self.config['persistence']['pca_filename']
            if pca_path.exists():
                self.pca = joblib.load(pca_path)
                logger.info(f"PCA loaded from {pca_path}")

        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
