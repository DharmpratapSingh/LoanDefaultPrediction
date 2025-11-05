"""Model training module for loan default prediction."""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Handles model training and management."""

    def __init__(self, config: dict):
        """
        Initialize the model trainer with configuration.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.models = {}

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'logistic'
    ) -> LogisticRegression:
        """
        Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'logistic' or 'lasso'

        Returns:
            Trained model
        """
        try:
            logger.info(f"Training {model_type} regression model")

            if model_type == 'lasso':
                params = self.config['models']['lasso_regression']
            else:
                params = self.config['models']['logistic_regression']

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            logger.info(f"{model_type.capitalize()} regression training completed")
            return model
        except Exception as e:
            logger.error(f"Error training {model_type} regression: {str(e)}")
            raise

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        try:
            logger.info("Training Random Forest model")

            params = self.config['models']['random_forest']
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            logger.info("Random Forest training completed")
            return model
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            raise

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> XGBClassifier:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        try:
            logger.info("Training XGBoost model")

            params = self.config['models']['xgboost']
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            logger.info("XGBoost training completed")
            return model
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            raise

    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> SVC:
        """
        Train SVM model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        try:
            logger.info("Training SVM model")

            params = self.config['models']['svm']
            model = SVC(**params)
            model.fit(X_train, y_train)

            logger.info("SVM training completed")
            return model
        except Exception as e:
            logger.error(f"Error training SVM: {str(e)}")
            raise

    def build_neural_network(self, input_dim: int) -> Sequential:
        """
        Build Neural Network architecture.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled model
        """
        try:
            logger.info(f"Building Neural Network with input_dim={input_dim}")

            params = self.config['models']['neural_network']

            model = Sequential([
                Dense(128, activation='relu', input_dim=input_dim),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(params['dropout_rate']),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            optimizer = Adam(learning_rate=params['learning_rate'])
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Neural Network architecture built successfully")
            return model
        except Exception as e:
            logger.error(f"Error building Neural Network: {str(e)}")
            raise

    def train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Sequential:
        """
        Train Neural Network model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Trained model
        """
        try:
            logger.info("Training Neural Network model")

            params = self.config['models']['neural_network']

            model = self.build_neural_network(X_train.shape[1])

            history = model.fit(
                X_train, y_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(X_test, y_test),
                verbose=0
            )

            logger.info(f"Neural Network training completed. Final loss: {history.history['loss'][-1]:.4f}")
            return model
        except Exception as e:
            logger.error(f"Error training Neural Network: {str(e)}")
            raise

    def train_stacking_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> StackingClassifier:
        """
        Train Stacking ensemble model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained stacking model
        """
        try:
            logger.info("Training Stacking ensemble model")

            # Define base models
            rf_params = self.config['models']['random_forest']
            xgb_params = self.config['models']['xgboost']

            estimators = [
                ('rf', RandomForestClassifier(**rf_params)),
                ('xgb', XGBClassifier(**xgb_params))
            ]

            # Create stacking classifier
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=3
            )

            stacking_model.fit(X_train, y_train)

            logger.info("Stacking model training completed")
            return stacking_model
        except Exception as e:
            logger.error(f"Error training Stacking model: {str(e)}")
            raise

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        include_neural_net: bool = True
    ) -> Dict[str, Any]:
        """
        Train all models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for neural network validation)
            y_test: Test labels (for neural network validation)
            include_neural_net: Whether to train neural network

        Returns:
            Dictionary of trained models
        """
        try:
            logger.info("Starting training for all models")

            models = {}

            # Logistic Regression
            models['logistic'] = self.train_logistic_regression(X_train, y_train, 'logistic')

            # Lasso Regression
            models['lasso'] = self.train_logistic_regression(X_train, y_train, 'lasso')

            # Random Forest
            models['random_forest'] = self.train_random_forest(X_train, y_train)

            # XGBoost
            models['xgboost'] = self.train_xgboost(X_train, y_train)

            # SVM
            models['svm'] = self.train_svm(X_train, y_train)

            # Stacking
            models['stacking'] = self.train_stacking_model(X_train, y_train)

            # Neural Network (optional - takes longer)
            if include_neural_net:
                models['neural_network'] = self.train_neural_network(
                    X_train, y_train, X_test, y_test
                )

            self.models = models
            logger.info(f"All models trained successfully. Total models: {len(models)}")

            return models
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            raise

    def save_model(
        self,
        model: Any,
        model_name: str,
        save_dir: str
    ):
        """
        Save a trained model to disk.

        Args:
            model: Trained model
            model_name: Name of the model
            save_dir: Directory to save the model
        """
        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            if isinstance(model, Sequential):
                # Save Keras model
                model_path = Path(save_dir) / f"{model_name}.h5"
                model.save(model_path)
            else:
                # Save scikit-learn or XGBoost model
                model_path = Path(save_dir) / f"{model_name}.joblib"
                joblib.dump(model, model_path)

            logger.info(f"Model '{model_name}' saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model '{model_name}': {str(e)}")
            raise

    def save_all_models(self, save_dir: Optional[str] = None):
        """
        Save all trained models.

        Args:
            save_dir: Directory to save models (uses config if not provided)
        """
        if save_dir is None:
            save_dir = self.config['persistence']['models_dir']

        try:
            logger.info(f"Saving all models to {save_dir}")

            for model_name, model in self.models.items():
                self.save_model(model, model_name, save_dir)

            logger.info(f"All models saved successfully to {save_dir}")
        except Exception as e:
            logger.error(f"Error saving all models: {str(e)}")
            raise

    def load_model(
        self,
        model_name: str,
        load_dir: str
    ) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_name: Name of the model
            load_dir: Directory containing the model

        Returns:
            Loaded model
        """
        try:
            # Try loading as Keras model first
            h5_path = Path(load_dir) / f"{model_name}.h5"
            if h5_path.exists():
                model = load_model(h5_path)
                logger.info(f"Model '{model_name}' loaded from {h5_path}")
                return model

            # Try loading as joblib model
            joblib_path = Path(load_dir) / f"{model_name}.joblib"
            if joblib_path.exists():
                model = joblib.load(joblib_path)
                logger.info(f"Model '{model_name}' loaded from {joblib_path}")
                return model

            raise FileNotFoundError(f"Model '{model_name}' not found in {load_dir}")

        except Exception as e:
            logger.error(f"Error loading model '{model_name}': {str(e)}")
            raise
