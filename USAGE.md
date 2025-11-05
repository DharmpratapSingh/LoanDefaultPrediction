# Usage Guide - Loan Default Prediction

This guide explains how to use the refactored Loan Default Prediction project.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Making Predictions](#making-predictions)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd LoanDefaultPrediction
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Training All Models
Train all models with default configuration:
```bash
python -m src.main --mode train
```

Train including neural network (takes longer):
```bash
python -m src.main --mode train --include-neural-net
```

Train only linear models:
```bash
python -m src.main --mode train_linear
```

Train both (all models + linear models without PCA):
```bash
python -m src.main --mode both
```

### Making Predictions

#### Single Prediction
```python
from src.inference import LoanDefaultPredictor

# Initialize predictor
predictor = LoanDefaultPredictor(model_name="random_forest")

# Create loan application data
loan_data = {
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

# Make prediction
result = predictor.predict_single(loan_data)
print(result)
# Output: {'default': False, 'default_probability': 0.23, 'risk_level': 'Low Risk'}
```

#### Batch Predictions
```python
from src.inference import LoanDefaultPredictor

predictor = LoanDefaultPredictor(model_name="random_forest")

# Process batch from CSV
predictions_df = predictor.predict_batch(
    csv_path="data/new_loans.csv",
    output_path="data/predictions.csv"
)
```

## Training Models

### Using Python API

```python
from src.main import train_pipeline
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Train models
models, results = train_pipeline(
    config_path='config/config.yaml',
    include_neural_net=True
)

# Access trained models
random_forest_model = models['random_forest']
xgboost_model = models['xgboost']
```

### Using Command Line

```bash
# Basic training
python -m src.main

# Custom configuration file
python -m src.main --config path/to/config.yaml

# Include neural network
python -m src.main --include-neural-net

# Train linear models only
python -m src.main --mode train_linear
```

## Configuration

Edit `config/config.yaml` to customize:

### Data Paths
```yaml
data:
  raw_data_path: "Loan prediction mini dataset.csv"
  processed_data_dir: "data/processed"
```

### Preprocessing Parameters
```yaml
preprocessing:
  test_size: 0.2
  random_state: 42
  age_cap: 100
```

### Model Parameters
```yaml
models:
  random_forest:
    n_estimators: 500
    max_depth: 30
    min_samples_split: 2
```

## Project Structure

```
LoanDefaultPrediction/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── __init__.py
│   ├── main.py                  # Main training pipeline
│   ├── data_preprocessing.py    # Data preprocessing module
│   ├── model_training.py        # Model training module
│   ├── evaluation.py            # Model evaluation module
│   ├── inference.py             # Inference pipeline
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging utilities
├── notebooks/
│   └── LoanDefaultPrediction.ipynb  # Original notebook
├── models/                      # Saved model artifacts
├── logs/                        # Training logs
├── reports/                     # Evaluation reports
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── USAGE.md                     # This file
```

## Advanced Usage

### Custom Preprocessing

```python
from src.data_preprocessing import DataPreprocessor
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

preprocessor = DataPreprocessor(config)

# Load and preprocess data
df = preprocessor.load_data("data/loans.csv")
df = preprocessor.handle_missing_values(df)
df = preprocessor.handle_outliers(df)
df = preprocessor.encode_target(df, 'Default')
```

### Custom Model Training

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(config)

# Train specific model
rf_model = trainer.train_random_forest(X_train, y_train)

# Save model
trainer.save_model(rf_model, "my_random_forest", "models/")
```

### Model Evaluation

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate single model
metrics = evaluator.evaluate_model(
    model=rf_model,
    X_test=X_test,
    y_test=y_test,
    model_name="Random Forest"
)

# Compare multiple models
results_df = evaluator.evaluate_all_models(models, X_test, y_test)

# Generate comprehensive report
evaluator.generate_report(models, X_test, y_test, output_dir="reports")
```

## Output Files

After training, you'll find:

- **models/** - Saved model files (.joblib, .h5)
- **logs/** - Training logs with timestamps
- **reports/** - Evaluation reports and plots
  - `model_evaluation_results.csv` - Metrics comparison
  - `model_comparison.png` - Visual comparison
  - `confusion_matrix_*.png` - Confusion matrices

## Troubleshooting

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/LoanDefaultPrediction
python -m src.main
```

### Memory Issues
If neural network training causes memory issues:
```bash
python -m src.main --mode train  # Without --include-neural-net
```

### CUDA/GPU Issues
For TensorFlow GPU issues, set environment variable:
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
python -m src.main --include-neural-net
```

## Examples

See `notebooks/LoanDefaultPrediction.ipynb` for the original exploratory analysis.

For more examples, check the `examples/` directory (coming soon).

## Support

For issues or questions, please open an issue on GitHub.
