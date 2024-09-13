# hypothesis_testing.py

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# Load the dataset
df = pd.read_csv('data/insurance_data.csv')

# Define functions for hypothesis testing

def test_gender_risk(df):
    """Test if there's a significant risk difference between genders."""
    male_claims = df[df['Gender'] == 'M']['TotalClaims']
    female_claims = df[df['Gender'] == 'F']['TotalClaims']
    t_stat, p_value = ttest_ind(male_claims, female_claims)
    print(f"P-value for risk differences between genders: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Significant risk differences between women and men.")
    else:
        print("Fail to reject the null hypothesis: No significant risk differences between women and men.")

def test_province_risk(df):
    """Test if there's a significant risk difference across provinces."""
    contingency_table = pd.crosstab(df['Province'], df['TotalClaims'])
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    print(f"P-value for risk differences across provinces: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Significant risk differences across provinces.")
    else:
        print("Fail to reject the null hypothesis: No significant risk differences across provinces.")

def test_zip_code_risk(df):
    """Test if there's a significant risk difference between zip codes."""
    contingency_table = pd.crosstab(df['ZipCode'], df['TotalClaims'])
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    print(f"P-value for risk differences between zip codes: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Significant risk differences between zip codes.")
    else:
        print("Fail to reject the null hypothesis: No significant risk differences between zip codes.")

def test_margin_difference(df):
    """Test if there's a significant margin (profit) difference between zip codes."""
    # Assuming margin data is available in the dataframe
    contingency_table = pd.crosstab(df['ZipCode'], df['Margin'])
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    print(f"P-value for margin differences between zip codes: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Significant margin differences between zip codes.")
    else:
        print("Fail to reject the null hypothesis: No significant margin differences between zip codes.")

# Run the tests
test_gender_risk(df)
test_province_risk(df)
test_zip_code_risk(df)
test_margin_difference(df)
