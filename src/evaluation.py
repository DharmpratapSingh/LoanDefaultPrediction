"""Model evaluation module for loan default prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Handles model evaluation and comparison."""

    def __init__(self):
        """Initialize the model evaluator."""
        self.results = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Evaluating model: {model_name}")

            # Get predictions
            if isinstance(model, Sequential):
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = None

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

            # Add ROC-AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC-AUC for {model_name}: {str(e)}")
                    metrics['roc_auc'] = None
            else:
                metrics['roc_auc'] = None

            # Log metrics
            logger.info(f"Metrics for {model_name}:")
            for metric, value in metrics.items():
                if value is not None:
                    logger.info(f"  {metric}: {value:.4f}")

            # Print detailed classification report
            logger.info(f"\nClassification Report for {model_name}:")
            logger.info("\n" + classification_report(y_test, y_pred))

            self.results[model_name] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise

    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate all models and return comparison DataFrame.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with all model metrics
        """
        try:
            logger.info("Evaluating all models")

            results = {}
            for model_name, model in models.items():
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                results[model_name] = metrics

            # Create comparison DataFrame
            df_results = pd.DataFrame(results).T
            df_results = df_results.round(4)

            logger.info("\nModel Comparison:")
            logger.info("\n" + df_results.to_string())

            return df_results

        except Exception as e:
            logger.error(f"Error evaluating all models: {str(e)}")
            raise

    def plot_confusion_matrix(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix for a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        try:
            logger.info(f"Plotting confusion matrix for {model_name}")

            # Get predictions
            if isinstance(model, Sequential):
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix for {model_name}: {str(e)}")
            raise

    def plot_model_comparison(
        self,
        df_results: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of all models.

        Args:
            df_results: DataFrame with model metrics
            save_path: Path to save the plot (optional)
        """
        try:
            logger.info("Plotting model comparison")

            # Prepare data
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            available_metrics = [m for m in metrics_to_plot if m in df_results.columns]

            models = df_results.index.tolist()
            x = np.arange(len(models))
            width = 0.15

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 7))

            for i, metric in enumerate(available_metrics):
                values = df_results[metric].fillna(0).values
                offset = width * (i - len(available_metrics) / 2)
                ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())

            # Customize plot
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.0)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {save_path}")

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")
            raise

    def get_best_model(
        self,
        df_results: pd.DataFrame,
        metric: str = 'roc_auc'
    ) -> str:
        """
        Get the name of the best model based on a metric.

        Args:
            df_results: DataFrame with model metrics
            metric: Metric to use for comparison

        Returns:
            Name of the best model
        """
        try:
            if metric not in df_results.columns:
                logger.warning(f"Metric '{metric}' not found. Using 'accuracy' instead.")
                metric = 'accuracy'

            best_model = df_results[metric].idxmax()
            best_score = df_results.loc[best_model, metric]

            logger.info(f"Best model based on {metric}: {best_model} (score: {best_score:.4f})")

            return best_model

        except Exception as e:
            logger.error(f"Error getting best model: {str(e)}")
            raise

    def save_results(
        self,
        df_results: pd.DataFrame,
        save_path: str
    ):
        """
        Save evaluation results to CSV.

        Args:
            df_results: DataFrame with model metrics
            save_path: Path to save the CSV file
        """
        try:
            df_results.to_csv(save_path)
            logger.info(f"Results saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def generate_report(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = "reports"
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report with plots.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save reports and plots

        Returns:
            DataFrame with evaluation results
        """
        try:
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            logger.info(f"Generating evaluation report in {output_dir}")

            # Evaluate all models
            df_results = self.evaluate_all_models(models, X_test, y_test)

            # Save results
            results_path = Path(output_dir) / "model_evaluation_results.csv"
            self.save_results(df_results, str(results_path))

            # Plot model comparison
            comparison_plot_path = Path(output_dir) / "model_comparison.png"
            self.plot_model_comparison(df_results, str(comparison_plot_path))

            # Plot confusion matrices for top models
            for model_name in df_results.index[:3]:  # Top 3 models
                if model_name in models:
                    cm_path = Path(output_dir) / f"confusion_matrix_{model_name}.png"
                    self.plot_confusion_matrix(
                        models[model_name],
                        X_test,
                        y_test,
                        model_name,
                        str(cm_path)
                    )

            # Get best model
            best_model = self.get_best_model(df_results)

            logger.info(f"Evaluation report generated successfully in {output_dir}")
            logger.info(f"Best performing model: {best_model}")

            return df_results

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
