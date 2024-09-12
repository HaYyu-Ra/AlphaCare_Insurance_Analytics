import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/historical_claims.csv')

# Data Summary
def data_summary(df):
    print("Data Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe())
    print("\nMissing Values:\n", df.isnull().sum())

# Univariate Analysis: Histograms
def plot_histograms(df, columns):
    for col in columns:
        plt.figure(figsize=(10,6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# Bivariate Analysis: Scatter Plot between TotalPremium and TotalClaims
def scatter_plot(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='Province', data=df)
    plt.title('Premium vs Claims by Province')
    plt.show()

if __name__ == "__main__":
    data_summary(df)
    plot_histograms(df, ['TotalPremium', 'TotalClaims'])
    scatter_plot(df)
