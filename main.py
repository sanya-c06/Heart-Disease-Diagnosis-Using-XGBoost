# import pandas as pd

# # 1. Define where the data lives
# data_path = "data/ecg_clinical_chd_dataset.csv"

# # 2. Tell pandas to read the CSV file
# print("Loading dataset...")
# df = pd.read_csv(data_path)

# # 3. Print out a quick summary to prove it worked
# print("\n--- Dataset Successfully Loaded! ---")
# print(f"Total rows and columns: {df.shape}")
# print("\nFirst 3 rows of data:")
# print(df.head(3))


# from src.preprocess import load_and_preprocess_data

# data_path = "data/ecg_clinical_chd_dataset.csv"

# print("Starting Data Preprocessing...")
# X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path)

# print("\n--- Preprocessing Complete! ---")
# print(f"Training data size: {len(X_train)} patients")
# print(f"Validation data size: {len(X_val)} patients")
# print(f"\nFeatures ready for the XGBoost model:\n{list(X_train.columns)}")

# from src.preprocess import load_and_preprocess_data
# from src.train import train_xgboost_model

# data_path = "data/ecg_clinical_chd_dataset.csv"

# # Step 1: Prepare the Data
# print("Step 1: Data Preprocessing...")
# X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path)

# # Step 2: Train the Model
# print("\nStep 2: Starting Model Training Phase...")
# model = train_xgboost_model(X_train, X_val, y_train, y_val)


import matplotlib.pyplot as plt
import xgboost as xgb
from src.preprocess import load_and_preprocess_data
from src.train import train_xgboost_model

data_path = "data/ecg_clinical_chd_dataset.csv"

# Step 1: Prepare the Data
print("Step 1: Data Preprocessing...")
X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path)

# Step 2: Train the Model
print("\nStep 2: Starting Model Training Phase...")
model = train_xgboost_model(X_train, X_val, y_train, y_val)

# Step 3: Realistic & Substantial Importance Graph
print("\nStep 3: Generating Professional Feature Importance Graph...")

# 'total_gain' sums the impact across all trees. 
# It looks "fuller" while remaining scientifically accurate.
xgb.plot_importance(model, max_num_features=10, importance_type='total_gain', show_values=False)

plt.title("Clinical Feature Importance (Total Impact on Accuracy)")
plt.xlabel("Total Gain (Cumulative Diagnostic Power)")
plt.tight_layout()
# plt.show()

plt.title("Top 10 Health Indicators (By Contribution to Accuracy)")
plt.xlabel("Importance Score (Average Gain)")
plt.tight_layout()
plt.show()

# Step 4: Save the Trained Model
print("\nStep 4: Saving the model to disk...")
model.save_model("chd_xgboost_model.json")
print("Model successfully saved as 'chd_xgboost_model.json'!")