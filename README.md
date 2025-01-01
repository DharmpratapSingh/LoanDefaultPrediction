# Loan Default Prediction Project

# Introduction

The project focuses on predicting whether a loan will default based on borrower attributes. This classification problem helps financial institutions assess risk and make informed decisions regarding loan approvals.

# Problem Statement

Loan default prediction is crucial for minimizing financial risks. The aim is to build a machine learning pipeline that:
	•	Predicts loan defaults with high precision and recall.
	•	Handles class imbalance effectively.
	•	Provides interpretability for decision-making.

Objectives
	1.	Preprocess the dataset to handle missing values, outliers, and categorical variables.
	2.	Address class imbalance using SMOTE.
	3.	Build and evaluate multiple machine learning models:
  	•	Logistic Regression
  	•	Lasso Regression
  	•	Random Forest
  	•	XGBoost
  	•	Neural Networks
  	•	SVM
	4.	Optimize the best models using hyperparameter tuning.
	5.	Compare model performance based on evaluation metrics.
	6.	Provide actionable insights through model interpretability.

# Dataset

Source: [Loan Prediction Dataset](https://www.kaggle.com/datasets/ganjerlawrence/loan-risk-prediction-dataset/data)

The dataset contains information on borrowers and their loans, with attributes such as:
	•	Numerical Columns: Age, Income, Loan Amount, etc.
	•	Categorical Columns: Home Ownership, Loan Purpose (Intent).
	•	Target Column: Default (Yes/No).

Key Statistics:
	•	Total Records: 8,145
	•	Class Distribution: Imbalanced (Default = N (majority), Y (minority)).

# Preprocessing Steps

1. Handling Missing Values
	•	Imputed Emp_length with the mode.
	•	Imputed Rate with the median.

2. Encoding Categorical Variables
	•	Applied one-hot encoding to Home and Intent columns:
	•	Dropped one category for linear models to avoid multicollinearity.

3. Addressing Class Imbalance
	•	Used SMOTE to oversample the minority class (Default = Y) to balance the dataset.

4. Scaling
	•	Applied Min-Max Scaling to numerical features to normalize their range.

5. Dimensionality Reduction
	•	Applied PCA to reduce dimensionality and retain 95% variance.

# Modeling and Evaluation

Trained Models:
	1.	Logistic Regression
	2.	Lasso Regression
	3.	Random Forest
	4.	XGBoost
	5.	Neural Networks
	6.	SVM

Evaluation Metrics:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
	•	ROC-AUC

Key Findings:
	•	Linear Models:
	•	Performed better with dataset_linear (avoiding multicollinearity).
	•	Logistic Regression achieved ROC-AUC = 0.92.
	•	Non-Linear Models:
	•	Random Forest and XGBoost outperformed other models.
	•	Random Forest: ROC-AUC = 0.92.
	•	XGBoost: ROC-AUC = 0.88.
	•	Neural Networks:
	•	Improved after tuning, achieving moderate performance.
	•	SVM:
	•	Underperformed compared to other models.

Hyperparameter Tuning

Random Forest:
	•	Used GridSearchCV to tune n_estimators, max_depth, and other parameters.
	•	Best ROC-AUC: 0.92.

XGBoost:
	•	Used RandomizedSearchCV to tune learning_rate, max_depth, subsample, and colsample_bytree.
	•	Best ROC-AUC: 0.88.

Neural Networks:
	•	Added dropout layers and batch normalization.
	•	Optimized learning rate, batch size, and epochs.

# Final Results and Insights

Model	              Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.86	    0.86	    0.86	  0.86	    0.92
Lasso Regression	  0.86	    0.87	    0.87	  0.86	    0.91
Random Forest	      0.84	    0.84	    0.84	  0.84	    0.92
XGBoost	            0.81	    0.82	    0.81	  0.81	    0.88
Neural Networks	    0.60	    0.60	    0.60	  0.60	    -
SVM	                0.57	    0.57	    0.57	  0.56	    0.60

Key Insights:
	1.	Random Forest and XGBoost are the best-performing models.
	2.	Logistic Regression performed well with dataset_linear, achieving similar ROC-AUC as Random Forest.
	3.	Neural Networks require further tuning and more data for competitive performance.

# Conclusion
	•	The Random Forest model is recommended for its balanced performance across all metrics.
	•	XGBoost is a strong alternative, especially for datasets with complex non-linear patterns.
	•	Linear models (Logistic, Lasso) are suitable when interpretability is prioritized.

# Future Work
	1.	Ensemble Models:
	  •	Combine Random Forest and XGBoost predictions for potential performance gains.
	2.	Feature Engineering:
	  •	Explore interaction terms and non-linear transformations.
	3.	Advanced Neural Networks:
	  •	Use deep learning frameworks for larger datasets with more features.
	4.	Explainability:
	  •	Implement SHAP or LIME to understand feature importance and model behavior.
