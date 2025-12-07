# Heart Disease Risk Assessment System


## 📌 Project Overview
This project is an end-to-date machine learning solution designed to predict the risk of heart disease based on the CDC's **Behavioral Risk Factor Surveillance System (BRFSS)** dataset. 

The system goes beyond simple prediction by integrating **K-Means clustering** for advanced feature engineering and **SHAP** (SHapley Additive exPlanations) for model interpretability. The final model is deployed via a **Streamlit** web application, allowing users to input their health metrics and receive a personalized risk assessment.

## 🗂️ Project Structure

| File | Description |
| :--- | :--- |
| `1.EDA.ipynb` | **Exploratory Data Analysis**: Deep dive into data distributions, correlations, and identifying key risk factors. |
| `2.Data Preprocessing.ipynb` | **Feature Engineering**: Handling missing values, scaling, and performing **K-Means Clustering** to create a 'Health Profile' feature. |
| `3. Model Training...ipynb` | **Modeling**: Training LightGBM/XGBoost models, hyperparameter tuning (Optuna), and SHAP interpretation. |
| `app.py` | **Deployment**: The interactive Streamlit dashboard source code. |
| `Heart_disease_model.pkl` | **Artifact**: The final trained LightGBM model file. |

## 📊 Model Performance

Given the medical nature of the problem, the dataset is highly imbalanced (few positive cases). Therefore, the modeling strategy prioritized **Recall** (Sensitivity) to minimize false negatives—ensuring that high-risk individuals are not missed.

We adjusted the classification threshold to **0.1** (instead of the default 0.5) to maximize disease detection.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8841** | Excellent discrimination capability between healthy and high-risk subjects. |
| **Recall (Positive Class)** | **0.88** | The model correctly identifies **88%** of actual heart disease cases. |
| **Precision (Positive Class)** | **0.13** | Due to the aggressive threshold (0.1) used to boost recall, precision is lower, meaning the model is cautious and may flag borderline cases for further checkups. |
| **Accuracy** | **0.65** | Overall accuracy across all samples. |

## 🛠️ Key Methodologies

1.  **Clustering-Based Feature Engineering**:
    * In `2.Data Preprocessing.ipynb`, we used **K-Means** to group individuals into health clusters. This cluster label was added as a feature, significantly improving model context.
2.  **Imbalanced Data Handling**:
    * Utilized **Threshold Tuning** rather than aggressive resampling to preserve the original data distribution while achieving high recall.
3.  **Explainable AI (XAI)**:
    * The app uses **SHAP values** to explain *why* a specific risk score was assigned (e.g., "Your risk is high primarily due to Age > 65 and BMI > 30").

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/heart-disease-risk-assessment.git](https://github.com/your-username/heart-disease-risk-assessment.git)
cd heart-disease-risk-assessment