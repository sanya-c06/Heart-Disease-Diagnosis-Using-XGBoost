# High-Precision Cardiovascular Disease Prediction using XGBoost

An advanced machine learning framework leveraging **XGBoost (Extreme Gradient Boosting)** alongside a multi-model comparative ensemble pipeline to predict Coronary Heart Disease (CHD) and Cardiovascular Diseases (CVD). This system achieves objective risk stratification by fusing patient clinical history with signal-based features extracted from Electrocardiogram (ECG) data via Intrinsic Mode Function (IMF) decomposition.

This repository focuses on the optimized XGBoost pipeline developed as part of a comparative framework at VIT Bhopal University.

---

## 📌 Project Overview
Conventional cardiovascular risk screenings rely heavily on manual interpretation, which introduces subjective diagnostic delays and elevates the risk of false negatives. This project implements an objective, automated pipeline using **XGBoost** to accurately map complex non-linear relationships between clinical parameters and oscillatory cardiac stress signals.

### Why XGBoost?
XGBoost is an optimized distributed gradient boosting library engineered for extreme speed and scalability. In a clinical deployment context, it provides exceptional throughput. By utilizing a sparsity-aware split-finding algorithm and parallel tree boosting, it handles missing data elements seamlessly. Furthermore, built-in $L_1$ (Lasso) and $L_2$ (Ridge) regularization weights prevent the model from overfitting to localized sensor noise or minor anomalies within raw ECG readings.

---

## 🛠️ Tech Stack
The project environment and pipeline are built entirely in **Python** using the following specialized tools:
- **Machine Learning Libraries:** XGBoost, CatBoost, Scikit-Learn (Random Forest baseline & performance metrics)
- **Deep Learning Frameworks:** PyTorch / Hugging Face (Attention-based Tab-Transformer deep learning model)
- **Data Engineering & Signal Processing:** Pandas, NumPy, SciPy, Seaborn & Matplotlib (for professional diagnostic plotting)
- **Core Pipeline Scripts:**
  - `preprocess.py`: Implements Z-score normalization for continuous metrics and handles feature mapping.
  - `train.py`: Sets up advanced tree parameters, early stopping, and data-driven class imbalance metrics.
  - `evaluate.py`: Generates multi-panel performance evaluation curves and confusion matrices.
  - `predict.py`: An end-to-end clinical simulation interface to evaluate single patient risk profiles.

---

## 🚀 Key Features
- **Tabular & Signal Feature Fusion:** Seamless hybrid integration of static metrics (Cholesterol, Blood Pressure, Age) and time-series fluctuations (ECG IMF components).
- **Advanced Regularization Control:** Tight restriction constraints via `reg_alpha=0.5` and `reg_lambda=1.5` to eliminate overfitting on clinical outliers.
- **Dynamic Imbalance Handling:** Auto-calculates an exact mathematical `scale_pos_weight` ratio to counter medical dataset skewness and boost recall sensitivity.
- **Early Stopping Optimization:** Monitors log-loss, classification error, and AUC over validation splits to halt training within 15 rounds of stagnation.

---

## 📊 Dataset & Feature Engineering
The models are trained and cross-validated on a combined dataset consisting of **2,500 patient records** (`ecg_clinical_chd_dataset.csv`):
1. **Clinical Features:** Static predictors including Age, Sex (One-Hot Encoded via `Sex_M`), Diabetes index, Smoking status, Blood Pressure (Systolic/Diastolic), and Serum Cholesterol levels.
2. **Signal Features:** Continuous raw ECG signals broken down into specific oscillatory patterns through **Intrinsic Mode Function (IMF) decomposition** (`IMF_Energy`, `IMF_Amplitude`, `IMF_Frequency`, and `HRV`).

**Data Preprocessing Pipeline (`preprocess.py`):**
- Unique identifier tags (`Patient_ID`) are discarded to protect privacy and eliminate indexing bias.
- Continuous clinical and signal variables are transformed via **StandardScaler Z-Score Normalization** to balance performance scales.

---

## 🔄 Model Comparison Framework
To evaluate which methodological approach best identifies non-linear dependencies across heterogeneous healthcare data, XGBoost was benchmarked alongside four other advanced architectures under a unified framework:

| Algorithmic Model | Core Methodology & Architectural Role | Advantages in This Framework | Limitations |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Sparsity-aware gradient boosting optimized via $L_1$/$L_2$ regularization weights and early stopping thresholds. | Rapid execution throughput speed; highly effective at managing missing clinical history inputs automatically. | Highly sensitive to outliers if hyperparameters aren't carefully tuned. |
| **CatBoost** | Gradient boosting utilizing symmetric trees to evaluate categorical relationships natively. | Eliminates prediction shift; zero manual encoding overhead; exceptional out-of-the-box robustness. | Slower initial training times relative to basic tree architectures. |
| **Random Forest** | Bagging-based ensemble that aggregates a broad forest of uncorrelated decision trees to set a baseline. | Immune to overfitting on small clinical variations; robust against noisy sensor data from ECG leads. | Struggles to map highly complex, non-linear deep signal waves. |
| **Transformer** | Tab-Transformer deep learning model using Multi-Head Self-Attention modules to track feature dependencies. | Captures simultaneous feature interactions (e.g., how Age cross-modulates with localized ECG frequency spikes). | Computationally expensive; requires extensive data scaling to converge. |
| **Stacking Ensemble** | Consensus meta-learning layer that uses base predictions from the above models as inputs for a meta-learner. | **Highest Performance:** Aggregates individual model weights to eliminate isolated biases and maximize performance. | Exploded architectural complexity; functions as an uninterpretable "black-box". |

### Summary of Comparative Results
- **Ensemble Dominance:** The **Stacking Ensemble** emerged as the top-performing model, achieving a peak **Accuracy of 95.4%** and a **Precision of 99.5%**.
- **XGBoost Operational Utility:** While the stacking model maximizes metric scores, **XGBoost** serves as the optimal standalone production model due to its low memory footprint and high clinical pipeline throughput speed.

---
**Author:** Sanya Chadha  
**University:** VIT Bhopal University  
**ID:** 24BCE10626