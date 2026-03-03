# 🚜 Bulldozer Price Prediction: Regression Analysis
> **Focus:** Random Forest Regression, Feature Engineering, and Error Minimization.

This project implements a supervised machine learning model to predict the sale price of heavy equipment. The primary objective was to build a robust **Random Forest Regressor** capable of handling large datasets with significant missing values and diverse categorical features.

## 🚀 The Challenge
The "Blue Book for Bulldozers" dataset contains over 50 variables, many of which are categorical or null. The core challenge was transforming this "messy" data into a numerical format that a machine learning model could process without losing the underlying predictive patterns.

## 🛠️ Technical Implementation
* **Frameworks:** Scikit-Learn, NumPy, Pandas, Matplotlib.
* **Algorithm:** Random Forest Regressor — chosen for its ability to handle non-linear relationships and high-dimensional data.
* **Preprocessing Pipeline:** * **Numerical Imputation:** Filled missing values using the median to maintain data distribution integrity.
    * **Categorical Encoding:** Converted string-based features into numerical categories to allow the model to interpret hardware specifications.
    * **Data Splitting:** Separated the dataset into training and validation sets to ensure the model could generalize to unseen auction data.

## 🎯 Results & Metrics
* **Evaluation Metric:** Root Mean Squared Log Error (RMSLE).
* **Significance:** RMSLE was used to ensure that an error in predicting a $10,000 bulldozer is treated with the same weight as an error for a $100,000 bulldozer, focusing on the quality of the prediction ratio.
* **Optimization:** Fine-tuned the Random Forest using `RandomizedSearchCV` to find the most efficient combination of `n_estimators` and `max_samples`.

## 📂 Project Workflow
1. **Exploratory Data Analysis (EDA):** Visualizing the distribution of sale prices and identifying key hardware features.
2. **Data Cleaning:** Handling the extensive null values present in the equipment specifications.
3. **Model Training:** Training a baseline Random Forest and iteratively improving its performance.
4. **Evaluation:** Validating the model's accuracy using the RMSLE metric on a dedicated validation set.
