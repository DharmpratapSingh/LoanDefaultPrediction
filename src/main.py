"""Main script for training and evaluating loan default prediction models."""

import argparse
import yaml
from pathlib import Path

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def train_pipeline(config_path: str = "config/config.yaml", include_neural_net: bool = False):
    """
    Execute the complete training pipeline.

    Args:
        config_path: Path to configuration file
        include_neural_net: Whether to train neural network model
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING LOAN DEFAULT PREDICTION TRAINING PIPELINE")
        logger.info("=" * 80)

        # Load configuration
        config = load_config(config_path)

        # Step 1: Data Preprocessing
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 80)

        preprocessor = DataPreprocessor(config)
        data_path = config['data']['raw_data_path']

        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            file_path=data_path,
            apply_pca_flag=True,
            for_linear_model=False
        )

        # Save preprocessors
        models_dir = config['persistence']['models_dir']
        preprocessor.save_preprocessors(models_dir)

        # Step 2: Model Training
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 80)

        trainer = ModelTrainer(config)
        models = trainer.train_all_models(
            X_train, y_train, X_test, y_test,
            include_neural_net=include_neural_net
        )

        # Save models
        trainer.save_all_models(models_dir)

        # Step 3: Model Evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("=" * 80)

        evaluator = ModelEvaluator()
        df_results = evaluator.generate_report(models, X_test, y_test, output_dir="reports")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nModels saved to: {models_dir}")
        logger.info(f"Evaluation reports saved to: reports/")
        logger.info(f"\nBest Model: {evaluator.get_best_model(df_results)}")

        return models, df_results

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


def train_linear_models_pipeline(config_path: str = "config/config.yaml"):
    """
    Execute training pipeline for linear models (without PCA, with column dropping).

    Args:
        config_path: Path to configuration file
    """
    try:
        logger.info("=" * 80)
        logger.info("TRAINING LINEAR MODELS PIPELINE")
        logger.info("=" * 80)

        # Load configuration
        config = load_config(config_path)

        # Data Preprocessing for linear models
        preprocessor = DataPreprocessor(config)
        data_path = config['data']['raw_data_path']

        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            file_path=data_path,
            apply_pca_flag=False,
            for_linear_model=True
        )

        # Train only linear models
        trainer = ModelTrainer(config)

        logger.info("Training linear models...")
        models = {
            'logistic_linear': trainer.train_logistic_regression(X_train, y_train, 'logistic'),
            'lasso_linear': trainer.train_logistic_regression(X_train, y_train, 'lasso')
        }

        # Save models
        models_dir = config['persistence']['models_dir']
        for model_name, model in models.items():
            trainer.save_model(model, model_name, models_dir)

        # Evaluate
        evaluator = ModelEvaluator()
        df_results = evaluator.evaluate_all_models(models, X_test, y_test)

        logger.info("\nLinear Models Results:")
        logger.info("\n" + df_results.to_string())

        return models, df_results

    except Exception as e:
        logger.error(f"Error in linear models pipeline: {str(e)}")
        raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Loan Default Prediction - Training and Evaluation Pipeline"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'train_linear', 'both'],
        default='train',
        help='Execution mode: train (all models), train_linear (linear models only), or both'
    )

    parser.add_argument(
        '--include-neural-net',
        action='store_true',
        help='Include neural network training (takes longer)'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'train':
            train_pipeline(args.config, args.include_neural_net)

        elif args.mode == 'train_linear':
            train_linear_models_pipeline(args.config)

        elif args.mode == 'both':
            # Train all models with PCA
            logger.info("Training all models with PCA...")
            train_pipeline(args.config, args.include_neural_net)

            # Train linear models without PCA
            logger.info("\n\nTraining linear models without PCA...")
            train_linear_models_pipeline(args.config)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
