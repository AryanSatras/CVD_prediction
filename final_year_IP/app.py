import streamlit as st
import pickle
import numpy as np


st.set_page_config(
    page_title="Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
)


#logic to load the pickle model
@st.cache_resource
def load_model():
    return pickle.load(open('final_ensemble_model.pkl', 'rb'))


model = load_model()
#headr
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>❤️ Cardiovascular Risk Predictor</h1>
        <p style='font-size:18px'>Predict the likelihood of cardiovascular disease using Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


input_col, result_col = st.columns([1, 1])

#user input column left
with input_col:
    st.subheader("📝 Patient Details")
    st.markdown("Fill out the patient information below:")

    # Group 1: Personal Info
    with st.expander("👤 Personal Info", expanded=True):
        age = st.number_input("Age", min_value=20, max_value=80, value=50,
                              help="Patient age in years")
        gender_dict = {'Male': 0, 'Female': 1}
        selected_gender = st.selectbox("Gender", list(gender_dict.keys()))
        gender = gender_dict[selected_gender]

    # Group 2: Vital Signs
    with st.expander("❤️ Vital Signs", expanded=False):
        restingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        serumcholestrol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fastbloodsugar_dict = {"No": 0, "Yes": 1}

        selected_fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            list(fastbloodsugar_dict.keys())
        )

        fastingbloodsugar = fastbloodsugar_dict[selected_fbs]
        maxheartrate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

    # Group 3: ECG & Exercise
    with st.expander("🏃 ECG & Exercise", expanded=False):
        chestpain = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3],
                                 help="0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic")
        restingrelectro = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
        exerciseangia = st.selectbox("Exercise Induced Angina", [0, 1])
        slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        noofmajorvessels = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])


    st.markdown("---")

#prediction column
with result_col:
    st.subheader("📊 Prediction Result")

    if st.button("🔍 Predict Risk"):
        import shap
        import matplotlib.pyplot as plt
        import pandas as pd

        gb_model = pickle.load(open('gb_model.pkl', 'rb'))

        features_names = ['age', 'gender', 'chestpain', 'retingBP', 'serumcholestrol','fastingbloodsugar','restingelecto', 'maxheartrate', 'exerciseangia', 'slope', 'oldpeak', 'noofmajorvessels']

        input_df = pd.DataFrame([[age, gender, chestpain, restingBP,
                                serumcholestrol, fastingbloodsugar,
                                restingrelectro, maxheartrate,
                                exerciseangia, oldpeak,
                                slope, noofmajorvessels]],columns=features_names)

        explainer = shap.TreeExplainer(gb_model)
        shap_values = explainer(input_df)

        st.subheader("Model explanation")
        shap.plots.waterfall(shap_values[0],show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # Prepare input
        input_data = np.array([[age, gender, chestpain, restingBP,
                                serumcholestrol, fastingbloodsugar,
                                restingrelectro, maxheartrate,
                                exerciseangia, oldpeak,
                                slope, noofmajorvessels]])
        # Prediction
        risk_prob = model.predict_proba(input_data)[0][1]
        risk_percent = round(risk_prob * 100, 2)

        # Risk progress bar with color
        if risk_percent < 30:
            st.success(f"🟢 Low Risk: {risk_percent}%")
            bar_color = "#4CAF50"  # Green
        elif risk_percent < 70:
            st.warning(f"🟡 Moderate Risk: {risk_percent}%")
            bar_color = "#FFC107"  # Yellow
        else:
            st.error(f"🔴 High Risk: {risk_percent}%")
            bar_color = "#F44336"  # Red

        # Custom progress bar
        st.markdown(
            f"""
            <div style="background-color:#eee; border-radius:10px; width:100%; height:25px;">
                <div style="width:{risk_percent}%; background-color:{bar_color}; height:100%; border-radius:10px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Extra info
        st.markdown(f"**Predicted Probability of Disease:** {risk_prob:.4f}")
        st.markdown("---")
        st.info(
            "⚠️ Note: This prediction is for educational purposes and not a substitute for professional medical advice.")
