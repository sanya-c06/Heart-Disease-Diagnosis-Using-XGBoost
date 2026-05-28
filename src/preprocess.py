import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Drop unique identifiers
    df = df.drop('Patient_ID', axis=1)

    # 3. Handle Categorical Data (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

    # 4. Standardize Numeric Features
    # We only scale the raw numbers, not the 0/1 labels
    numeric_cols = [
        'IMF_Energy', 'IMF_Amplitude', 'IMF_Frequency', 'HRV', 
        'BP_Systolic', 'BP_Diastolic', 'Age', 'Cholesterol'
    ]
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 5. Split into Features (X) and Target (y)
    X = df.drop('CHD_Label', axis=1)
    y = df['CHD_Label']

    # 6. Train-Validation Split (80% Train, 20% Val)
    return train_test_split(X, y, test_size=0.2, random_state=42)