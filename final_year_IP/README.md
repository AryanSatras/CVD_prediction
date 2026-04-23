#  Cardiovascular Disease Prediction System

This project is a machine learning-based system designed to predict the risk of cardiovascular disease (CVD) using patient data. The system combines multiple models using an ensemble approach and integrates Explainable AI (XAI) to provide interpretable predictions.

---

##  Features

- Predicts cardiovascular disease risk based on user input
- Uses an ensemble model for improved accuracy
- Provides real-time predictions via a Streamlit interface
- Includes Explainable AI using SHAP for feature-level interpretation
- Displays visual explanation of how each feature affects the prediction

---

##  Models Used

- Logistic Regression 
- Decision Tree
- KNN
- Random Forest  
- Gradient Boosting  
- Ensemble Model (final prediction)

Gradient Boosting is used for SHAP-based explanation.

---

## 🖥️ User Interface

The application is built using Streamlit and allows users to:
- Enter patient details
- View predicted risk percentage
- See visual explanation of prediction (SHAP waterfall plot)

---

## streamlits Explainable AI

This project uses SHAP (SHapley Additive exPlanations) to:
- Explain model predictions
- Show how each feature contributes to risk
- Improve transparency and trust in the system

---

##  Project Structure
# Cardiovasular_disease_prediction
final_year_IP/
│── app.py # Main Streamlit app
│── main.py # Supporting script
│── final_ensemble_model.pkl # Ensemble model
│── gb_model.pkl # Gradient Boosting model (for SHAP)
│── cardio_rf_model.pkl # Random Forest model
│── requirements.txt
│── Cardiovascular_Disease_Dataset.csv(main dataset)
│── *.ipynb # Model development notebooks
│── README.md

## ⚙️ Installation

1. Clone the repository:
```bash
git clone <https://github.com/AryanSatras/Cardiovasular_disease_prediction.git>
cd final_year_IP

#install dependencies packages 
pip install -r requirements.txt

How it Works
User enters patient details
Ensemble model predicts risk
Gradient Boosting model explains prediction using SHAP
Results are displayed with visual explanation


Limitations
Model trained on limited dataset
Not a substitute for medical diagnosis
SHAP explanations may require basic understanding

Note
This project is developed for educational purposes as part of a Final Year Project.
