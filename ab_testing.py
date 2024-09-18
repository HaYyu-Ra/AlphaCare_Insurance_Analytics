import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# Load Data Function
def load_data():
    """
    Load the insurance claims data from the specified path.
    :return: DataFrame containing the data
    """
    data_path = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/insurance_claims.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file at {data_path} does not exist.")
    
    try:
        # Load the dataset
        data = pd.read_csv(data_path, encoding='utf-8')
        
        # Check if the DataFrame is empty
        if data.empty:
            raise ValueError("The file is empty.")
        
        # Print column names and the first few rows for debugging
        print("Column names:", data.columns)
        print("Data preview:\n", data.head())
        
        return data
    
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file. The file might be empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the file. The file might be corrupted or not a valid CSV.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

# Data Cleaning Function
def clean_data(data):
    """
    Perform data cleaning tasks such as handling missing values, correcting data types,
    removing duplicates, removing empty columns and rows, and filtering invalid data.
    
    :param data: Pandas DataFrame
    :return: Cleaned DataFrame
    """
    # Remove empty columns
    data = data.dropna(axis=1, how='all')
    
    # Remove rows with all NaN values
    data = data.dropna(axis=0, how='all')
    
    # Drop duplicates
    data = data.drop_duplicates()
    
    # Handling missing values - Example: Fill missing values in 'TotalClaims' with mean
    if 'TotalClaims' in data.columns:
        data['TotalClaims'] = data['TotalClaims'].fillna(data['TotalClaims'].mean())
    
    # Convert 'Zipcode' to string (if it's a categorical feature)
    if 'Zipcode' in data.columns:
        data['Zipcode'] = data['Zipcode'].astype(str)
    
    # Remove rows where 'TotalClaims' or 'ProfitMargin' are negative or invalid
    if 'TotalClaims' in data.columns and 'ProfitMargin' in data.columns:
        data = data[(data['TotalClaims'] >= 0) & (data['ProfitMargin'] >= 0)]
    
    # Ensure 'HighRisk' column is binary
    if 'HighRisk' in data.columns:
        data['HighRisk'] = data['HighRisk'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return data

# Hypothesis Testing Functions
def t_test(data, group_col, target_col, group1, group2):
    """
    Perform a t-test between two groups in a numerical column.
    
    :param data: Pandas DataFrame
    :param group_col: Column used to divide into groups (categorical)
    :param target_col: Numerical column to compare between the groups
    :param group1: Name of the first group
    :param group2: Name of the second group
    :return: t-statistic and p-value from the t-test
    """
    try:
        group1_data = data[data[group_col] == group1][target_col]
        group2_data = data[data[group_col] == group2][target_col]

        if group1_data.empty or group2_data.empty:
            raise ValueError("One or both groups are empty.")
        
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, nan_policy='omit')
        
        return t_stat, p_value
    except KeyError as e:
        raise KeyError(f"Column not found in the data: {e}")
    except ValueError as e:
        raise ValueError(f"Error in t-test: {e}")

def chi_square_test(data, col1, col2):
    """
    Perform a chi-square test of independence between two categorical variables.
    
    :param data: Pandas DataFrame
    :param col1: First categorical column
    :param col2: Second categorical column
    :return: chi-square statistic and p-value
    """
    try:
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return chi2_stat, p_value
    except KeyError as e:
        raise KeyError(f"Column not found in the data: {e}")

def run_ab_testing(data):
    """
    Run a series of A/B hypothesis tests and display results.
    """
    st.header("A/B Hypothesis Testing")
    
    # Example 1: Test for risk differences by province
    st.subheader("1. Risk Differences by Province")
    st.write("Null Hypothesis: There are no risk differences across provinces.")
    
    if 'Province' in data.columns and 'TotalClaims' in data.columns:
        st.write("Performing t-test for risk differences by Province...")
        t_stat, p_value = t_test(data, 'Province', 'TotalClaims', 'Province_A', 'Province_B')  
        st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("Result: Reject the null hypothesis (significant difference in risk between provinces).")
        else:
            st.write("Result: Fail to reject the null hypothesis (no significant difference).")
    else:
        st.write("Required columns are missing for this test.")
    
    # Example 2: Test for profit margin differences by zipcode
    st.subheader("2. Profit Margin Differences by Zipcode")
    st.write("Null Hypothesis: There are no profit margin differences across zipcodes.")
    
    if 'Zipcode' in data.columns and 'ProfitMargin' in data.columns:
        st.write("Performing t-test for profit margin differences by Zipcode...")
        t_stat, p_value = t_test(data, 'Zipcode', 'ProfitMargin', 'Zipcode_1', 'Zipcode_2')  
        st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("Result: Reject the null hypothesis (significant difference in profit margin across zipcodes).")
        else:
            st.write("Result: Fail to reject the null hypothesis.")
    else:
        st.write("Required columns are missing for this test.")
    
    # Example 3: Test for risk differences between genders
    st.subheader("3. Risk Differences by Gender")
    st.write("Null Hypothesis: There are no risk differences between genders.")
    
    if 'Gender' in data.columns and 'HighRisk' in data.columns:
        st.write("Performing chi-square test for risk differences by Gender...")
        chi2_stat, p_value = chi_square_test(data, 'Gender', 'HighRisk')  
        st.write(f"Chi-square statistic: {chi2_stat:.4f}, P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.write("Result: Reject the null hypothesis (significant association between gender and risk).")
        else:
            st.write("Result: Fail to reject the null hypothesis (no significant association).")
    else:
        st.write("Required columns are missing for this test.")

def visualize_ab_testing_results(data):
    """
    Visualize the results of A/B testing using boxplots or bar charts.
    """
    st.header("A/B Testing Visualizations")
    
    # Example: Boxplot comparing Total Claims by Province
    if 'Province' in data.columns and 'TotalClaims' in data.columns:
        st.subheader("Boxplot: Total Claims by Province")
        fig, ax = plt.subplots()
        sns.boxplot(x='Province', y='TotalClaims', data=data, ax=ax)
        st.pyplot(fig)
    
    # Example: Barplot comparing Profit Margin by Zipcode
    if 'Zipcode' in data.columns and 'ProfitMargin' in data.columns:
        st.subheader("Barplot: Profit Margin by Zipcode")
        fig, ax = plt.subplots()
        sns.barplot(x='Zipcode', y='ProfitMargin', data=data, ax=ax)
        st.pyplot(fig)

# Main function to run the A/B testing pipeline
def main():
    # Load data
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Clean data
    data = clean_data(data)
    
    # Run A/B testing
    run_ab_testing(data)
    
    # Visualize results
    visualize_ab_testing_results(data)

if __name__ == "__main__":
    main()
