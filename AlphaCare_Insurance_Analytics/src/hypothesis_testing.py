import pandas as pd
from scipy import stats

# Load dataset
df = pd.read_csv('data/historical_claims.csv')

# Define null hypothesis tests
def hypothesis_test(group_col, value_col):
    groups = df[group_col].unique()
    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f'Test between {groups[0]} and {groups[1]}: p-value = {p_value}')
    if p_value < 0.05:
        print("Reject Null Hypothesis")
    else:
        print("Fail to Reject Null Hypothesis")

# Apply tests for different hypotheses
if __name__ == "__main__":
    # Risk difference across provinces
    hypothesis_test('Province', 'TotalClaims')
    # Risk difference between genders
    hypothesis_test('Gender', 'TotalClaims')
    # Profit margin difference between zip codes
    hypothesis_test('PostalCode', 'TotalPremium')
