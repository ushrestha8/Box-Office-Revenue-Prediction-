🎬 Box Office Revenue Prediction using Machine Learning

Project Overview:
This project implements a machine learning solution to predict a movie's domestic box office revenue prior to its release. 
Using historical data, we employ data cleaning, extensive feature engineering, and the robust XGBoost Regressor to build a strong predictive model and identify the primary drivers of commercial success.

Key Objectives

Predictive Model: Train an XGBoost model to accurately estimate expected domestic revenue.Feature Importance: Provide clear insights into which features (e.g., MPAA rating, genres, theater count) most influence revenue.

⚙️ Methodology and Pipeline

The prediction workflow follows standard ML best practices:

  1. Data Preprocessing
       Cleaning: Removed special characters ($, ,) and converted features like domestic_revenue, opening_theaters, and release_days to numeric format.
       Imputation: Handled missing values in categorical features (MPAA, genres) using mode imputation.
       Feature Removal: Dropped irrelevant or highly missing columns (world_revenue, opening_revenue, budget).
    
2. Feature Engineering
     Log Transformation: Applied $\log_{10}$ transformation to all numeric features (including the target $\mathbf{domestic\_revenue}$) to reduce extreme skewness and stabilize the model.

    Categorical Encoding:
        Genres: Used CountVectorizer to transform the genre text field into a sparse set of binary features (one-hot encoding).
        MPAA & Distributor: Used LabelEncoder to convert these into ordinal integers.Normalization: Applied StandardScaler to all features prior to modeling.
   
4. Model Training & Evaluation
   Data Split: $90\%$ Training data, $10\%$ Validation data.

   Model: XGBoost Regressor (XGBRegressor) was trained on the processed data.

   Metric: Mean Absolute Error (MAE) was used to measure prediction accuracy.

   MetricTraining SetValidation SetMAE (Log Scale)$0.210$$0.636$

🛠️ Requirements

To run the Python script locally, you will need:

    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    
How to Run

    Clone the repository and ensure all three files are present.
    Execute the script from your terminal
