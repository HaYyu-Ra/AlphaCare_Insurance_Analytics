import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Path to the dataset
DATA_PATH = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/MachineLearningRating_v3.txt'

def extract_data(file_path):
    """Extract the data from the given file path."""
    try:
        # Load the dataset
        print("Loading data from:", file_path)
        data = pd.read_csv(file_path, sep='|', low_memory=False)  # Assuming '|' is the delimiter
        print("Data loaded successfully. Number of rows and columns:", data.shape)
        return data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def transform_data(data):
    """Clean and transform the data."""
    
    print("\n--- Starting Data Cleaning and Transformation ---")
    
    # Drop rows with all NaN values
    data.dropna(how='all', inplace=True)
    
    # Handling missing values
    print("Handling missing values...")
    data.fillna(value={
        'RegistrationYear': data['RegistrationYear'].median(),
        'SumInsured': data['SumInsured'].median(),
        'CalculatedPremiumPerTerm': data['CalculatedPremiumPerTerm'].mean()
    }, inplace=True)
    
    # Convert data types
    print("Converting data types...")
    data['RegistrationYear'] = pd.to_numeric(data['RegistrationYear'], errors='coerce')
    data['SumInsured'] = pd.to_numeric(data['SumInsured'], errors='coerce')
    data['CalculatedPremiumPerTerm'] = pd.to_numeric(data['CalculatedPremiumPerTerm'], errors='coerce')

    # Convert date columns to datetime
    if 'VehicleIntroDate' in data.columns:
        data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], errors='coerce')
        data['VehicleIntroYear'] = data['VehicleIntroDate'].dt.year
        data['VehicleIntroMonth'] = data['VehicleIntroDate'].dt.month
        data['VehicleIntroDay'] = data['VehicleIntroDate'].dt.day
        data.drop(columns=['VehicleIntroDate'], inplace=True)
    
    # Remove duplicates
    print("Removing duplicate records...")
    data.drop_duplicates(inplace=True)

    # Additional data transformation logic (if any)
    # Example: Filtering rows based on conditions
    data = data[data['RegistrationYear'] >= 1990]  # Keeping records from 1990 and later

    # Encode categorical features
    print("Encoding categorical features...")
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    print("\n--- Data Cleaning Completed ---")
    print("Cleaned data shape:", data.shape)
    
    return data

def load_data(data, output_path):
    """Save the cleaned data to a new file."""
    print("\n--- Loading Data ---")
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def split_data(data):
    """Split the dataset into training and testing sets."""
    
    # Print column names for debugging
    print("Columns in dataset:", data.columns.tolist())
    
    # Correct target column (e.g., 'TotalClaims')
    target_column = 'TotalClaims'  # Replace with actual target column name

    # Check if the target column exists
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train machine learning models."""
    print("\n--- Training RandomForest Regressor Model ---")
    model = RandomForestRegressor(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    print("\n--- Evaluating Model ---")
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

def main():
    # Step 1: Extract
    data = extract_data(DATA_PATH)
    
    if data is not None:
        # Step 2: Transform
        cleaned_data = transform_data(data)
        
        # Step 3: Load
        OUTPUT_PATH = 'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/AlphaCare_Insurance_Analytics/data/cleaned_MachineLearningRating_v3.csv'
        load_data(cleaned_data, OUTPUT_PATH)
        
        # Step 4: Split the data
        X_train, X_test, y_train, y_test = split_data(cleaned_data)
        
        # Step 5: Train the model
        model = train_models(X_train, y_train)
        
        # Step 6: Evaluate the model
        evaluate_model(model, X_test, y_test)
    else:
        print("Failed to load the data.")

if __name__ == "__main__":
    main()
