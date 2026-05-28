import xgboost as xgb

def train_xgboost_model(X_train, X_val, y_train, y_val):
    # Calculate scale_pos_weight to handle unbalanced data automatically
    # Formula: (Number of Negative Cases) / (Number of Positive Cases)
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    imbalance_ratio = neg_cases / pos_cases

    # Advanced XGBoost Configuration based on Research Parameters
    model = xgb.XGBClassifier(
        n_estimators=1000,           # Set very high, Early Stopping will control it
        learning_rate=0.01,          # Slow learning rate for better accuracy
        max_depth=6,                 # Depth of trees
        subsample=0.8,               # Use 80% of data per tree to prevent overfitting
        colsample_bytree=0.8,        # Use 80% of features per tree
        reg_alpha=0.5,               # L1 Regularization (New!)
        reg_lambda=1.5,              # L2 Regularization (New!)
        scale_pos_weight=imbalance_ratio, # Fixes the 0.85 Recall issue (New!)
        early_stopping_rounds=15,    # Stops if validation doesn't improve for 15 rounds (New!)
        eval_metric=["logloss", "error", "auc"], # Added AUC for better medical evaluation
        random_state=42
    )

    # Train the model
    # Notice we pass the validation set here so early stopping can monitor it
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True # Set to True so you can watch the Early Stopping in action in your terminal!
    )
    
    return model