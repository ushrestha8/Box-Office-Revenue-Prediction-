🎬 Box Office Revenue Prediction:

An XGBoost Regression Project

✨ Project Summary

This project develops a machine learning model to accurately estimate a movie's domestic box office revenue. We focus on comprehensive data cleaning and feature engineering—including a critical $\log_{10}$ transformation to handle revenue skewness—before training a high-performance XGBoost Regressor.

The goal is to deliver both a strong predictive tool and clear insights into the factors (like MPAA rating and genre) that drive commercial success.

🎯 Key Methodologies

Log Transformation: Applied $\log_{10}$ to domestic_revenue, opening_theaters, and release_days to normalize highly skewed financial data, a necessity for stable regression modeling.

Genre Encoding: Used CountVectorizer to convert the text genres column into quantifiable, binary features for the model.

Categorical Handling: LabelEncoder was used on MPAA and distributor, followed by StandardScaler on all final features.

Feature Correlation: Used a heatmap (data exported as correlation_matrix.csv) to identify and manage multicollinearity among features.

🚀 Getting Started

Prerequisites

To run the analysis locally, ensure you have Python 3.9+ and the following libraries installed:

    Bash

    pip install pandas numpy scikit-learn matplotlib seaborn xgboost

Execution

Clone the repository:

    Bash

    git clone https://github.com/your-username/box-office-prediction.git
    cd box-office-prediction
    
Execute the main script:

    Bash

    python index.py
    
This script generates the model, prints performance metrics, and creates the required visualization file.
