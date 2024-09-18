import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. ETL: Load Data from TXT file
def load_data():
    """
    Load the insurance claims dataset from a .txt file.
    
    :return: Pandas DataFrame containing the data
    """
    # Data path (Update if necessary)
    data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\AlphaCare_Insurance_Analytics\data\MachineLearningRating_v3.txt"
    
    try:
        # Assuming the data is tab-separated
        data = pd.read_csv(data_path, delimiter="\t")
        if data.empty:
            raise ValueError("The dataset is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {data_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or cannot be read.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

# 2. Handle Missing Values
def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    :param data: Pandas DataFrame
    :return: Cleaned DataFrame with missing values handled
    """
    # Fill missing numerical values with the median
    num_cols = data.select_dtypes(include=np.number).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    
    # Fill missing categorical values with the mode
    cat_cols = data.select_dtypes(include='object').columns
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
    
    return data

# 3. Encode Categorical Variables
def encode_categorical_data(data):
    """
    Encode categorical variables using label encoding or one-hot encoding.
    
    :param data: Pandas DataFrame
    :return: DataFrame with categorical variables encoded
    """
    label_enc = LabelEncoder()
    
    # Label encode binary categorical columns
    binary_cols = [col for col in data.columns if data[col].nunique() == 2]
    for col in binary_cols:
        data[col] = label_enc.fit_transform(data[col])
    
    # One-Hot Encoding for multi-class categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    return data

# 4. Scale Numerical Features
def scale_numerical_features(data):
    """
    Scale numerical features using standard scaling (z-score normalization).
    
    :param data: Pandas DataFrame
    :return: Scaled DataFrame
    """
    scaler = StandardScaler()
    num_cols = data.select_dtypes(include=np.number).columns
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return data

# 5. Split the Data into Train and Test Sets
def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    :param data: Pandas DataFrame
    :param target_column: Column name of the target variable
    :param test_size: Proportion of the data to be used for testing
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Main Data Preprocessing Pipeline
def preprocess_data():
    """
    Full preprocessing pipeline to load, clean, encode, scale, and split data.
    
    :return: X_train, X_test, y_train, y_test
    """
    # Step 1: Load data
    data = load_data()
    
    # Step 2: Handle missing values
    data = handle_missing_values(data)
    
    # Step 3: Encode categorical variables
    data = encode_categorical_data(data)
    
    # Step 4: Scale numerical features
    data = scale_numerical_features(data)
    
    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target_column='Target')  # Replace 'Target' with actual column name
    
    return X_train, X_test, y_train, y_test

# Test Preprocessing Pipeline
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = preprocess_data()
        print("Training Data Shape:", X_train.shape)
        print("Testing Data Shape:", X_test.shape)
    except Exception as e:
        print(f"An error occurred: {e}")
