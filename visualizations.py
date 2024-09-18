import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Directory Path
DATA_PATH = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/MachineLearningRating_v3.txt'

# Load Data Function
def load_data():
    """
    Load data from the specified file path.
    """
    # Read the data file
    data = pd.read_csv(DATA_PATH, sep='\t', engine='python')
    return data

# Visualization Functions
def visualize_data(data):
    """
    Display various visualizations on the dashboard.
    """
    st.header("Data Visualizations")

    # 1. Distribution of Numerical Features
    st.subheader("Distribution of Numerical Features")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        st.write(f"Histogram of {col}:")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax, bins=30)
        st.pyplot(fig)
    
    # 2. Boxplots for Numerical Features by Categorical Variables
    st.subheader("Boxplots of Numerical Features by Categorical Variables")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in numerical_columns:
        for cat_col in categorical_columns:
            if cat_col in data.columns and len(data[cat_col].unique()) < 10:  # Only plot if categorical has <10 unique values
                st.write(f"Boxplot of {col} by {cat_col}:")
                fig, ax = plt.subplots()
                sns.boxplot(x=cat_col, y=col, data=data, ax=ax)
                st.pyplot(fig)

    # 3. Pairplot of Numerical Variables
    st.subheader("Pairplot of Numerical Variables")
    if len(data) <= 1000:  # Adjust based on dataset size
        fig = sns.pairplot(data[numerical_columns])
        st.pyplot(fig)
    else:
        st.write("Pairplot is disabled due to dataset size.")

    # 4. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if numerical_columns:
        fig, ax = plt.subplots()
        sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns available for correlation heatmap.")

    # 5. Count Plot for Categorical Variables
    st.subheader("Count Plot for Categorical Variables")
    for col in categorical_columns:
        st.write(f"Count plot of {col}:")
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=data, ax=ax)
        st.pyplot(fig)

    # 6. Scatter Plots
    st.subheader("Scatter Plots")
    for i in range(len(numerical_columns)):
        for j in range(i+1, len(numerical_columns)):
            col1 = numerical_columns[i]
            col2 = numerical_columns[j]
            st.write(f"Scatter plot between {col1} and {col2}:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[col1], y=data[col2], ax=ax)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    # Load data
    data = load_data()

    # Perform Data Visualizations
    visualize_data(data)

if __name__ == "__main__":
    main()
