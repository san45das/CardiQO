import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Trained model loading
try:
    model = joblib.load("heart_disease_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file 'heart_disease_model.pkl' not found.")
    st.stop()

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Display names
feature_name_mapping = {
    "age": "Age",
    "sex": "Sex",
    "chest pain type": "Chest Pain Type",
    "resting bp s": "Resting Blood Pressure",
    "cholesterol": "Cholesterol",
    "fasting blood sugar": "Fasting Blood Sugar",
    "resting ecg": "Resting ECG",
    "max heart rate": "Maximum Heart Rate",
    "exercise angina": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "ST slope": "ST Slope",
}

# NHS-based advice
health_advice = {
    "low_risk": [
        "👍 Keep up the good lifestyle habits!",
        "🥦 Continue eating balanced meals.",
        "🏃 Stay physically active.",
        "🧘 Manage stress through mindfulness or relaxation.",
        "🔗 Learn more: https://www.nhs.uk/conditions/coronary-heart-disease/prevention/",
    ],
    "high_risk": [
        "📞 Consider consulting a cardiologist soon.",
        "🥗 Adopt a heart-friendly diet (low salt & fat).",
        "🚭 If you smoke, seek help to quit.",
        "🏃‍♂️ Begin moderate physical activity (with approval).",
        "🔗 Learn more: https://www.nhs.uk/conditions/coronary-heart-disease/",
    ],
}


# Header & reset
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;'>
        <div style='flex-grow: 1; text-align: left;'>
            <h1 style='font-size: 3em; color: #FF4B4B; margin-bottom: 0;'>🫀 CardiQO: <span style="color: white;">Heart Disease Predictor</span></h1>
            <p style='font-size: 1.25em; color: #BBBBBB; margin-top: 4px;'>
                <i>Optimizing your heart with intelligent precision.</i>
            </p>
        </div>
        <form action="" method="get">
            <button style='margin-top: 15px; padding: 6px 12px; border-radius: 5px; background-color: #444; color: white; border: none;'>Reset</button>
        </form>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inputs
def user_input():
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-anginal Pain",
            4: "Asymptomatic",
        }[x],
    )
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    rest_ecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x],
    )
    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    ex_angina = st.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox(
        "ST Slope",
        [1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
    )

    return pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "chest pain type": cp,
                "resting bp s": resting_bp,
                "cholesterol": cholesterol,
                "fasting blood sugar": fbs,
                "resting ecg": rest_ecg,
                "max heart rate": max_hr,
                "exercise angina": ex_angina,
                "oldpeak": oldpeak,
                "ST slope": st_slope,
            }
        ]
    )

# SHAP explanation
def base_col_from_ohe(fname: str, input_df_raw: pd.DataFrame) -> str:
    """Map a transformed feature name (e.g., 'cat__ST slope_3') back to the original column."""
    tail = fname.split("__", 1)[1] if "__" in fname else fname
    
    candidates = [c for c in input_df_raw.columns if tail.startswith(c)]
    if candidates:
        return max(candidates, key=len)
    if "_" in tail:
        return tail.rsplit("_", 1)[0]
    return tail

def human_value(col: str, raw_val):
    if col == "ST slope":
        return {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(int(round(raw_val)), "?")
    if col == "chest pain type":
        return {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-anginal Pain",
            4: "Asymptomatic",
        }.get(int(round(raw_val)), "?")
    if col == "resting ecg":
        return {
            0: "Normal",
            1: "ST-T Abnormality",
            2: "Left Ventricular Hypertrophy",
        }.get(int(round(raw_val)), "?")
    if col == "sex":
        return "Male" if int(round(raw_val)) == 1 else "Female"
    if col == "exercise angina":
        return "Yes" if int(round(raw_val)) == 1 else "No"
    if col == "fasting blood sugar":
        return "High (>120 mg/dl)" if int(round(raw_val)) == 1 else "Normal (≤120 mg/dl)"
    return raw_val

def display_name(col: str) -> str:
    return feature_name_mapping.get(col, col.replace("_", " ").capitalize())

def interpret_shap(shap_contribs, feature_names, input_df_raw, predicted_class):
    """
    Make explanations consistent with the final prediction label:
      - If predicted_class == 1 (Likely HD): always say 'Increased risk of heart disease'
      - If predicted_class == 0 (Unlikely HD): always say 'Lowered risk of heart disease'
    Also deduplicate by original feature (avoid repeated ST slope lines).
    """
    abs_vals = np.abs(shap_contribs)
    order = np.argsort(abs_vals)[::-1]  # descending
    seen_cols = set()
    insights = []

    effect_phrase = (
        "⬆️ Increased risk of heart disease"
        if predicted_class == 1
        else "⬇️ Lowered risk of heart disease"
    )

    for idx in order:
        fname = feature_names[idx]
        base_col = base_col_from_ohe(fname, input_df_raw)

        if base_col in seen_cols:
            continue
        if base_col not in input_df_raw.columns:
            continue

        seen_cols.add(base_col)

        raw_val = input_df_raw.iloc[0][base_col]
        readable_val = human_value(base_col, raw_val)
        insights.append(
            f"{effect_phrase} due to **{display_name(base_col)}**, which is **{readable_val}**"
        )

        if len(insights) == 3:
            break

    return insights

# Main flow
input_df = user_input()

if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Likely Heart Disease (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Unlikely to Have Heart Disease (Confidence: {prob:.2f})")

    # SHAP
    raw_model = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(input_df)
    explainer = shap.Explainer(raw_model, X_transformed)
    shap_expl = explainer(X_transformed, check_additivity=False)[0]

    if shap_expl.values.ndim == 2 and shap_expl.values.shape[1] == 2:
        shap_vals = shap_expl.values[:, prediction]
    else:
        shap_vals = shap_expl.values

    feature_names = preprocessor.get_feature_names_out()

    # Explanation
    st.markdown("### 🧠 Why this prediction?")
    insights = interpret_shap(shap_vals, feature_names, input_df, prediction)
    for line in insights:
        st.markdown(f"- {line}")

    # Recommendations
    st.markdown("### 💡 What can you do?")
    advice_key = "high_risk" if prediction == 1 else "low_risk"
    for tip in health_advice[advice_key]:
        st.write(tip)

    # SHAP bar chart 
    if np.any(np.abs(shap_vals) > 1e-6):
        st.markdown("### 📊 SHAP Feature Impact")
        # Top 10 absolute values
        idx_sorted = np.argsort(np.abs(shap_vals))[::-1][:10]

        labels = []
        values = []
        for i in idx_sorted:
            fname = feature_names[i]
            base_col = base_col_from_ohe(fname, input_df)
            if base_col in input_df.columns:
                raw_v = human_value(base_col, input_df.iloc[0][base_col])
                dn = display_name(base_col)
                labels.append(f"{dn}: {raw_v}")
                values.append(shap_vals[i])

        if values:
            shap_exp = shap.Explanation(
                values=np.array(values),
                feature_names=labels,
                data=np.zeros_like(values),
            )
            shap.plots.bar(shap_exp, max_display=10, show=False)
            st.pyplot(plt.gcf())


