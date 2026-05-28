import time
# Start the timer!
start_time = time.time()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# Added precision_score, recall_score, and f1_score to imports
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_score, 
                            recall_score, f1_score)
from src.preprocess import load_and_preprocess_data

# 1. Load Data
X_train, X_val, y_train, y_val = load_and_preprocess_data("data/ecg_clinical_chd_dataset.csv")

# Calculate imbalance ratio to match our advanced training setup
neg_cases = (y_train == 0).sum()
pos_cases = (y_train == 1).sum()
imbalance_ratio = neg_cases / pos_cases

# 2. Train Model (Using the Advanced Research Parameters!)
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    scale_pos_weight=imbalance_ratio,
    early_stopping_rounds=15,
    random_state=42,
    eval_metric=["logloss", "error", "auc"] 
)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

# 3. Extract results for Accuracy & Loss Graphs
results = model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

train_accuracy = [1 - x for x in results['validation_0']['error']]
val_accuracy = [1 - x for x in results['validation_1']['error']]

# 4. Create Predictions 
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Stop the timer!
end_time = time.time()
execution_time = end_time - start_time

# --- NEW: Variable Declarations ---
# Get the exact best scores from the early stopping mechanism
best_iteration = model.best_iteration
best_auc = results['validation_1']['auc'][best_iteration]
best_acc = 1 - results['validation_1']['error'][best_iteration]

# Calculate Precision, Recall, and F1 based on the positive class (CHD)
best_precision = precision_score(y_val, y_pred)
best_recall = recall_score(y_val, y_pred)
best_f1 = f1_score(y_val, y_pred)

# 5. Print the Text Report (Formatted as requested)
print("=========================================")
print("   CHD PREDICTION MODEL PERFORMANCE      ")
print("=========================================")
print(f"Accuracy:       {best_acc * 100:.2f}%")
print(f"Precision:      {best_precision * 100:.2f}%")
print(f"Recall:         {best_recall * 100:.2f}%")
print(f"F1 Score:       {best_f1 * 100:.2f}%")
print(f"AUC Score:      {best_auc * 100:.2f}%")
print(f"Execution Time: {execution_time:.2f} seconds")
print("=========================================")

# ==========================================
# VISUALIZATIONS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1: Model Accuracy
axes[0, 0].plot(x_axis, train_accuracy, label='Train Acc', color='#1f77b4')
axes[0, 0].plot(x_axis, val_accuracy, label='Val Acc', color='#ff7f0e')
axes[0, 0].set_title('Model Accuracy Curve - XGBoost', fontsize=14)
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# Graph 2: Model Loss
axes[0, 1].plot(x_axis, results['validation_0']['logloss'], label='Train Loss')
axes[0, 1].plot(x_axis, results['validation_1']['logloss'], label='Val Loss')
axes[0, 1].set_title('Model Loss Curve - XGBoost', fontsize=14)
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# Graph 3: Professional Confusion Matrix (DNN Styling - Square Fix)
cm = confusion_matrix(y_val, y_pred)
labels = ['Healthy', 'CHD']

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    ax=axes[1, 0], 
    cbar=True,
    cbar_kws={"shrink": 0.8}, 
    xticklabels=labels, 
    yticklabels=labels, 
    annot_kws={"size": 14, "weight": "bold"},
    linewidths=1, 
    linecolor='white',
    square=True 
)

axes[1, 0].set_title('Confusion Matrix - XGBoost', fontsize=14, pad=20)
axes[1, 0].set_xlabel('Predicted Label', fontsize=12, labelpad=10)
axes[1, 0].set_ylabel('True Label', fontsize=12, labelpad=10)

# Graph 4: ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
axes[1, 1].plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {auc(fpr, tpr):.2f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[1, 1].set_title('ROC Curve - XGBoost', fontsize=14)
axes[1, 1].legend(loc="lower right")

# Final Layout Fixes
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.subplots_adjust(wspace=0.3, hspace=0.6) 
fig.suptitle("XGBoost Model Performance: Cardiovascular Disease Prediction", fontsize=16, y=0.98)

print("Evaluation Complete. Displaying Graphs...")
plt.show()