import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# Define the data path
data_path = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/MachineLearningRating_v3.txt'

# Load the data with tab delimiter
df = pd.read_csv(data_path, delimiter='|')

# Display the first few rows of the dataframe
print("Initial Data:")
print(df.head())

# Data Cleaning and Transformation
# Drop rows with missing values in critical columns
df.dropna(subset=['TotalPremium', 'TotalClaims', 'Province', 'PostalCode', 'Gender'], inplace=True)

# Convert columns to appropriate types
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')

# Check column names
print("Column Names:")
print(df.columns)

# Task 3: A/B Hypothesis Testing

# Hypothesis 1: Risk differences across provinces
provinces = df['Province'].unique()
group1 = df[df['Province'] == provinces[0]]['TotalClaims']
group2 = df[df['Province'] == provinces[1]]['TotalClaims']

# Perform t-test for risk differences across provinces
t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
print(f"\nRisk differences across provinces:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis 2: Risk differences between zip codes
zip_codes = df['PostalCode'].unique()
group1 = df[df['PostalCode'] == zip_codes[0]]['TotalClaims']
group2 = df[df['PostalCode'] == zip_codes[1]]['TotalClaims']

# Perform t-test for risk differences between zip codes
t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
print(f"\nRisk differences between zip codes:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis 3: Margin (profit) differences between zip codes
group1 = df[df['PostalCode'] == zip_codes[0]]['TotalPremium']
group2 = df[df['PostalCode'] == zip_codes[1]]['TotalPremium']

# Perform t-test for margin differences between zip codes
t_stat, p_value = ttest_ind(group1, group2, nan_policy='omit')
print(f"\nMargin differences between zip codes:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Hypothesis 4: Risk differences between Women and Men
male_claims = df[df['Gender'] == 'M']['TotalClaims']
female_claims = df[df['Gender'] == 'F']['TotalClaims']

# Perform t-test for risk differences between genders
t_stat, p_value = ttest_ind(male_claims, female_claims, nan_policy='omit')
print(f"\nRisk differences between Women and Men:")
print(f"T-statistic: {t_stat}, P-value: {p_value}")

# Interpret results based on p-values
def interpret_p_value(p_value):
    if p_value < 0.05:
        return "Reject the null hypothesis: There is a significant difference."
    else:
        return "Fail to reject the null hypothesis: There is no significant difference."

print("\nInterpretation:")
print("1. Risk differences across provinces:", interpret_p_value(p_value))
print("2. Risk differences between zip codes:", interpret_p_value(p_value))
print("3. Margin differences between zip codes:", interpret_p_value(p_value))
print("4. Risk differences between Women and Men:", interpret_p_value(p_value))
