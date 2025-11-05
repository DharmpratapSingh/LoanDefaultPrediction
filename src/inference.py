"""Inference pipeline for loan default prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LoanDefaultPredictor:
    """Production-ready inference pipeline for loan default prediction."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        model_name: str = "random_forest"
    ):
        """
        Initialize the predictor.

        Args:
            config_path: Path to configuration file
            model_name: Name of the model to use for predictions
        """
        self.config_path = config_path
        self.model_name = model_name
        self.config = None
        self.preprocessor = None
        self.model = None
        self.model_trainer = None

        logger.info(f"Initializing LoanDefaultPredictor with model: {model_name}")

    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def load_artifacts(self):
        """Load trained model and preprocessors."""
        try:
            if self.config is None:
                self.load_config()

            # Initialize components
            self.preprocessor = DataPreprocessor(self.config)
            self.model_trainer = ModelTrainer(self.config)

            # Load preprocessors
            models_dir = self.config['persistence']['models_dir']
            self.preprocessor.load_preprocessors(models_dir)
            logger.info("Preprocessors loaded successfully")

            # Load model
            self.model = self.model_trainer.load_model(self.model_name, models_dir)
            logger.info(f"Model '{self.model_name}' loaded successfully")

        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise

    def preprocess_input(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            data: Input data as DataFrame or dictionary

        Returns:
            Preprocessed features ready for prediction
        """
        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])

            logger.info(f"Preprocessing input data with shape: {data.shape}")

            # Handle missing values
            data = self.preprocessor.handle_missing_values(data)

            # Handle outliers
            data = self.preprocessor.handle_outliers(data)

            # Encode categorical variables
            cat_cols = self.config['preprocessing']['categorical_columns']
            existing_cat_cols = [col for col in cat_cols if col in data.columns]

            if existing_cat_cols:
                data = self.preprocessor.encode_categorical(
                    data, existing_cat_cols, drop_first=False
                )

            # Ensure all expected columns are present
            # This is important when one-hot encoding might create different columns
            # For production, you'd want to store the expected columns during training

            # Remove ID if present
            id_col = self.config['preprocessing']['id_column']
            if id_col in data.columns:
                data = data.drop(id_col, axis=1)

            # Scale numerical features
            num_cols = [
                col for col in self.config['preprocessing']['numerical_columns']
                if col in data.columns
            ]

            if num_cols:
                data = self.preprocessor.scale_features(
                    data, num_cols, fit=False
                )

            # Apply PCA if the preprocessor has it
            if self.preprocessor.pca is not None:
                data = self.preprocessor.apply_pca(data.values, n_components=None, fit=False)

            logger.info(f"Data preprocessed successfully. Shape: {data.shape if isinstance(data, np.ndarray) else data.shape}")

            return data

        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions on input data.

        Args:
            data: Input data as DataFrame or dictionary
            return_proba: Whether to return probabilities instead of classes

        Returns:
            Predictions (and probabilities if requested)
        """
        try:
            # Load artifacts if not already loaded
            if self.model is None:
                self.load_artifacts()

            # Preprocess input
            X = self.preprocess_input(data)

            # Make prediction
            if return_proba:
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict(X)
                    probabilities = self.model.predict_proba(X)
                    logger.info(f"Predictions generated with probabilities for {len(X)} samples")
                    return {
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
                else:
                    # For neural networks
                    probabilities = self.model.predict(X, verbose=0).flatten()
                    predictions = (probabilities > 0.5).astype(int)
                    logger.info(f"Predictions generated with probabilities for {len(X)} samples")
                    return {
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
            else:
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(X)
                else:
                    # For neural networks
                    probabilities = self.model.predict(X, verbose=0).flatten()
                    predictions = (probabilities > 0.5).astype(int)

                logger.info(f"Predictions generated for {len(predictions)} samples")
                return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_single(
        self,
        loan_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make prediction for a single loan application.

        Args:
            loan_data: Dictionary with loan application data

        Returns:
            Dictionary with prediction result and probability
        """
        try:
            logger.info("Making prediction for single loan application")

            result = self.predict(loan_data, return_proba=True)

            if isinstance(result, dict):
                prediction = int(result['predictions'][0])
                if isinstance(result['probabilities'], np.ndarray) and result['probabilities'].ndim > 1:
                    probability = float(result['probabilities'][0][1])
                else:
                    probability = float(result['probabilities'][0])
            else:
                prediction = int(result[0])
                probability = None

            output = {
                'default': bool(prediction),
                'default_probability': probability,
                'risk_level': self._categorize_risk(probability) if probability is not None else 'Unknown'
            }

            logger.info(f"Prediction: {output}")
            return output

        except Exception as e:
            logger.error(f"Error predicting single loan: {str(e)}")
            raise

    def _categorize_risk(self, probability: float) -> str:
        """
        Categorize loan risk based on default probability.

        Args:
            probability: Default probability

        Returns:
            Risk category
        """
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"

    def predict_batch(
        self,
        csv_path: str,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of loan applications from CSV.

        Args:
            csv_path: Path to CSV file with loan applications
            output_path: Path to save predictions (optional)

        Returns:
            DataFrame with predictions
        """
        try:
            logger.info(f"Making batch predictions from {csv_path}")

            # Load data
            data = pd.read_csv(csv_path)
            original_data = data.copy()

            # Make predictions
            result = self.predict(data, return_proba=True)

            # Add predictions to original data
            if isinstance(result, dict):
                original_data['prediction'] = result['predictions']
                if isinstance(result['probabilities'], np.ndarray) and result['probabilities'].ndim > 1:
                    original_data['default_probability'] = result['probabilities'][:, 1]
                else:
                    original_data['default_probability'] = result['probabilities']
                original_data['risk_level'] = original_data['default_probability'].apply(
                    self._categorize_risk
                )
            else:
                original_data['prediction'] = result

            # Save if output path provided
            if output_path:
                original_data.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to {output_path}")

            logger.info(f"Batch predictions completed for {len(original_data)} samples")
            return original_data

        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise


def create_sample_input() -> Dict[str, Any]:
    """
    Create a sample input for testing the inference pipeline.

    Returns:
        Sample loan application data
    """
    return {
        'Id': 12345,
        'Age': 30,
        'Income': 65000,
        'Home': 'MORTGAGE',
        'Emp_length': 5.0,
        'Intent': 'EDUCATION',
        'Amount': 10000,
        'Rate': 10.5,
        'Status': 0,
        'Percent_income': 0.15,
        'Cred_length': 5
    }


if __name__ == "__main__":
    # Example usage
    predictor = LoanDefaultPredictor(model_name="random_forest")

    # Single prediction
    sample_loan = create_sample_input()
    result = predictor.predict_single(sample_loan)
    print("\nSingle Prediction Result:")
    print(result)

    # Batch prediction example
    # predictor.predict_batch("path/to/loans.csv", "path/to/predictions.csv")
