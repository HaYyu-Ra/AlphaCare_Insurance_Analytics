import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np
import shap

# Directory Path
DATA_PATH = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/MachineLearningRating_v3.txt'

# Load Data Function
def load_data():
    """
    Load data from the specified file path.
    """
    try:
        data = pd.read_csv(DATA_PATH, sep='|', engine='python')
        if data.empty:
            st.warning("The dataset is empty.")
        return data
    except FileNotFoundError:
        st.error(f"File not found at path: {DATA_PATH}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("The file is empty or cannot be read.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

# Data Cleaning Function
def clean_data(data):
    """
    Perform data cleaning operations including handling missing values, data type conversion, and feature engineering.
    """
    if data.empty:
        return data

    # Drop columns with too many missing values
    missing_threshold = 0.5
    data = data.loc[:, data.isnull().mean() < missing_threshold]

    # Fill missing values for categorical columns with the mode
    for column in data.select_dtypes(include=['object']).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # Fill missing values for numerical columns with the median
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        data[column].fillna(data[column].median(), inplace=True)

    # Convert categorical columns to numeric using label encoding
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column].astype(str))

    # Add a new feature: Profit Margin
    if 'TotalPremium' in data.columns and 'SumInsured' in data.columns:
        data['ProfitMargin'] = data['TotalPremium'] - data['SumInsured']
    
    return data

# Task 3: A/B Hypothesis Testing
def ab_hypothesis_testing(data):
    st.header("A/B Hypothesis Testing")
    
    if data.empty:
        st.warning("No data available for A/B testing.")
        return

    # Select Metrics for testing
    st.subheader("Select Metrics for Hypothesis Testing")
    metric = st.selectbox("Choose a metric:", ["TotalClaims", "ProfitMargin", "TotalPremium"])

    # Data Segmentation
    st.subheader("Data Segmentation")
    group_a = st.selectbox("Group A (Control Group):", ["Province", "PostalCode", "Gender"])
    group_b = st.selectbox("Group B (Test Group):", ["Province", "PostalCode", "Gender"])

    # Perform statistical test
    st.subheader("Statistical Testing")
    st.write("Testing for statistical significance between the two groups.")
    group_a_data = data[data[group_a].notnull()]
    group_b_data = data[data[group_b].notnull()]

    # Chi-squared or t-test based on the data type
    if data[metric].dtype in ['int64', 'float64']:
        # Perform t-test for numerical data
        stat, p_value = stats.ttest_ind(group_a_data[metric], group_b_data[metric])
    else:
        # Perform chi-squared test for categorical data
        contingency_table = pd.crosstab(group_a_data[metric], group_b_data[metric])
        stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    st.write(f"Test Statistic: {stat}")
    st.write(f"P-Value: {p_value}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis: There is a statistically significant difference between the groups.")
    else:
        st.write("Fail to reject the null hypothesis: No significant difference between the groups.")
    
    st.subheader("Analysis and Report")
    st.write("Interpret the results and provide business insights based on hypothesis testing outcomes.")

# Task 4: Statistical Modeling
def statistical_modeling(data):
    st.header("Statistical Modeling")

    if data.empty:
        st.warning("No data available for modeling.")
        return

    # Data Preparation: Train-Test Split
    st.subheader("Data Preparation")
    feature_columns = st.multiselect("Select feature columns for prediction:", data.columns.tolist(), default=['TotalPremium'])
    target_column = st.selectbox("Select target variable:", ['TotalClaims', 'TotalPremium'])

    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Selection
    model_type = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest", "XGBoost"])

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor()
    else:
        model = xgb.XGBRegressor()

    # Model Training
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation
    st.subheader("Model Evaluation")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    # Feature Importance (For Tree-based Models)
    if model_type != "Linear Regression":
        st.subheader("Feature Importance")
        feature_importances = model.feature_importances_
        st.bar_chart(pd.Series(feature_importances, index=feature_columns))

    # SHAP Analysis
    st.subheader("SHAP Analysis")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Use Matplotlib figure for SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False)
    st.pyplot(fig)

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    st.header("Exploratory Data Analysis")

    if data.empty:
        st.warning("No data available for EDA.")
        return

    st.subheader("Data Overview")
    st.write(data.describe())

    st.subheader("Missing Values")
    missing_values = data.isnull().sum().sort_values(ascending=False)
    st.bar_chart(missing_values)

    st.subheader("Correlation Analysis")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Total Claims")
    fig, ax = plt.subplots()
    sns.histplot(data['TotalClaims'], kde=True, ax=ax)
    st.pyplot(fig)

# Main Streamlit App
def main():
    st.title("AlphaCare Insurance Analytics Dashboard")
    
    # Load and clean data
    data = load_data()
    data = clean_data(data)
    
    # Exploratory Data Analysis (EDA)
    perform_eda(data)
    
    # Task 3: A/B Hypothesis Testing
    ab_hypothesis_testing(data)
    
    # Task 4: Statistical Modeling
    statistical_modeling(data)

# Run the app
if __name__ == "__main__":
    main()
