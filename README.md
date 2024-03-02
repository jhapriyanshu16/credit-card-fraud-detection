# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used for this project is sourced from Kaggle, containing anonymized credit card transactions labeled as fraudulent or non-fraudulent.

## Dataset
- **Name:** Credit Card Fraud Detection
- **Link:** [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Overview
The goal of this project is to build a classification model that accurately identifies fraudulent transactions based on various features such as transaction amount, time, and other anonymized features. The project includes data preprocessing, handling imbalanced data, and evaluating multiple machine learning algorithms.

## Implementation Steps
1. **Data Loading and Exploration:** The dataset is loaded into a pandas DataFrame for exploration. Basic data statistics and information are analyzed.
2. **Data Preprocessing:** Null values are checked and feature scaling is performed on the 'Amount' column using StandardScaler. The 'Time' column is dropped as it's not relevant for model training.
3. **Handling Imbalanced Dataset:**
   - Undersampling: Random undersampling of majority class instances is performed to balance the dataset.
   - Oversampling: Synthetic Minority Over-sampling Technique (SMOTE) is used to oversample the minority class instances.
4. **Model Training and Evaluation:**
   - Logistic Regression, Decision Tree, and Random Forest classifiers are trained on both balanced datasets.
   - Model performance metrics such as accuracy, precision, recall, and F1-score are calculated for each model.
5. **Model Saving:** The trained Random Forest classifier is saved using joblib for future use.
6. **Prediction:** A sample transaction is used to demonstrate model prediction. If the prediction is 0, it indicates a normal transaction; otherwise, it's flagged as a fraudulent transaction.

## Libraries Used
- pandas
- scikit-learn
- joblib
- matplotlib


## How to Use
1. Clone the repository to your local machine.
2. Download the Credit Card Fraud Detection dataset from Kaggle.
3. Install the required libraries using the provided command.
4. Run the Jupyter Notebook to execute the project code.
5. Explore the model performance and predictions based on the provided example.

## Library Installation
```bash
pip install pandas scikit-learn joblib matplotlib

