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
    
    :return: Pandas DataFrame containing the data
    """
    try:
        # Read the data file
        data = pd.read_csv(DATA_PATH, sep='\t', engine='python')
        if data.empty:
            raise ValueError("The dataset is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {DATA_PATH}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or cannot be read.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

# Exploratory Data Analysis (EDA) section
def perform_eda(data):
    """
    Display Exploratory Data Analysis (EDA) section on the dashboard.
    Includes data summary, visualizations, and statistical analysis.
    
    :param data: Pandas DataFrame
    """
    st.header("Exploratory Data Analysis (EDA)")

    # 1. Display general statistics
    st.subheader("Dataset Overview")
    st.write("The dataset contains the following columns:")
    st.write(data.columns.tolist())  # List column names
    st.write(data.describe(include='all'))  # Summary statistics for all columns
    
    # 2. Show missing values count
    st.subheader("Missing Values")
    missing_data = data.isnull().sum().sort_values(ascending=False)
    st.write(missing_data)

    # 3. Distribution of categorical variables
    st.subheader("Categorical Variables Distribution")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        st.write(f"Distribution of {col}:")
        st.write(data[col].value_counts())

    # 4. Visualization: Distribution of Numerical Columns
    st.subheader("Distribution of Numerical Features")
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        st.write(f"Histogram of {col}:")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax, bins=30)
        st.pyplot(fig)
    
    # 5. Boxplots for Numerical Features by Categorical Variables
    st.subheader("Boxplots of Numerical Features by Categorical Variables")
    for col in numerical_columns:
        for cat_col in categorical_columns:
            if cat_col in data.columns and len(data[cat_col].unique()) < 10:  # Only plot if categorical has <10 unique values
                st.write(f"Boxplot of {col} by {cat_col}:")
                fig, ax = plt.subplots()
                sns.boxplot(x=cat_col, y=col, data=data, ax=ax)
                st.pyplot(fig)

    # 6. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if numerical_columns.size > 0:
        fig, ax = plt.subplots()
        sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns available for correlation heatmap.")

    # 7. Pairplot for Numerical Variables (if dataset is small)
    st.subheader("Pairplot of Numerical Variables")
    if len(data) <= 1000:  # Adjust based on dataset size
        if numerical_columns.size > 0:
            fig = sns.pairplot(data[numerical_columns])
            st.pyplot(fig)
        else:
            st.write("No numerical columns available for pairplot.")
    else:
        st.write("Pairplot is disabled due to dataset size.")

# Main function to run the Streamlit app
def main():
    # Load data
    data = load_data()

    # Perform Exploratory Data Analysis
    perform_eda(data)

if __name__ == "__main__":
    main()
