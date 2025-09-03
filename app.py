# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import textwrap


st.set_page_config(
    page_title="Thyroid Cancer Recurrence Prediction",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    label[data-baseweb="label"] {
        font-size: 20px !important;
        font-weight: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------

# ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    model          = joblib.load("stacking_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    features       = joblib.load("features.pkl")
    X_train        = joblib.load("X_train.pkl")
    return model, label_encoders, features, X_train

model, label_encoders, features, X_train = load_artifacts()

# ------------------------------------------------
# 2. 表单默认值
# ------------------------------------------------
default_values = {
    'Age': 56,
    'Gender': "F",
    'Smoking': "No",
    'Hx Smoking': "Yes",
    'Hx Radiothreapy': "No",
    'Thyroid Function': "Clinical Hyperthyroidism",
    'Physical Examination': 'Diffuse goiter',
    'Adenopathy': "Bilateral",
    'Pathology': "Papillary",
    'Focality': 'Multi-Focal',
    'Risk': "Low",
    'T': "T2",
    'N': "N1b",
    'M': "M0",
    'Stage': "II",
    'Response': 'Structural Incomplete'
}

# ------------------------------------------------

# ------------------------------------------------
st.markdown(
    "<h2 style='text-align: center; font-size: 34px;color: #2c3e50;'>"
    "🧠 Thyroid Cancer Recurrence Prediction System</h2>",
    unsafe_allow_html=True
)


st.markdown("""
<div style='text-align: justify; color: #34495e; font-size: 18px; line-height: 1.4;'>
The stacking-based prediction system integrated six base learners (DT, RF, LGBM, SVM, SGD, KNN) with LR as the meta-learner, aiming to predict thyroid cancer recurrence risk based on patient-specific features through artificial intelligence.
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------

# ------------------------------------------------
with st.form("prediction_form"):
    st.markdown("#### 🧾 Input Patient Features")
    cols = st.columns(3)
    user_input = {}

    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            if feature in label_encoders:
                options = label_encoders[feature].classes_.tolist()
                default_index = options.index(default_values.get(feature, options[0]))
                user_input[feature] = st.selectbox(feature, options, index=default_index)
            elif feature == "Age":
                user_input[feature] = st.number_input(
                    "Age", min_value=0, max_value=120,
                    value=int(default_values.get("Age", 50))
                )
            else:
                user_input[feature] = st.number_input(
                    feature, value=float(default_values.get(feature, 0.0))
                )

    submitted = st.form_submit_button("✅ Predict")

# ------------------------------------------------

# ------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input])

    
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    
    pred_prob = model.predict_proba(input_df)[0]
    pred_class = model.predict(input_df)[0]
    inv_label = label_encoders["Recurred"].inverse_transform([pred_class])[0]

    
    st.markdown("### 🎯 Prediction Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"🧬 Predicted Class: **{inv_label}**")
    with col2:
        st.info(f"📊 Probability: No Recurrence `{pred_prob[0]:.2f}` | Recurrence `{pred_prob[1]:.2f}`")

    recurrence_prob = pred_prob[1]
    st.markdown(
        f"""
        <div style='padding-top:10px; font-size:22px; color:#2c3e50; text-align: center;'>
        Based on feature values, predicted possibility of TC recurrence is <b>{recurrence_prob*100:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    with st.expander("🔍 Show SHAP Explanation (Class: Recurrence)"):
        explainer = shap.Explainer(model.predict_proba, X_train)
        shap_values = explainer(input_df)

        st.markdown("**📌 SHAP Force Plot**", unsafe_allow_html=True)
        st_shap(
            shap.plots.force(
                base_value=shap_values.base_values[0][1],
                shap_values=shap_values.values[0][:, 1],
                features=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            height=80
        )

        st.markdown("**📊 SHAP Waterfall Plot**", unsafe_allow_html=True)
        left, mid, right = st.columns([1, 10, 1])
        with mid:
            
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": [  "DejaVu Serif","Liberation Serif","Times","Times New Roman"],
                "font.size": 18,
                "axes.titlesize": 20,
                "axes.labelsize": 18,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
            })

            max_len = 28
            exp = shap.Explanation(
                values=np.round(shap_values.values[0, :, 1], 8),
                base_values=round(shap_values.base_values[0, 1], 8),
                data=input_df.iloc[0].values,
                feature_names=[
                    textwrap.shorten(f, width=max_len, placeholder="…")
                    for f in input_df.columns
                ],
            )

            fig = plt.figure(figsize=(5, 1.5))
            shap.plots.waterfall(exp, max_display=len(features), show=False)

            
            ax = fig.gca()
            for label in ax.get_yticklabels():
                label.set_fontname("Times New Roman")
                label.set_fontsize(18)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        
        st.markdown("**📌 Encoding and representation of clinical features**", unsafe_allow_html=True)
        mapping_list = []
        for feature in features:
            if feature not in label_encoders:
                mapping_list.append([feature, "None"])
        for col, le in label_encoders.items():
            if col != "Recurred":
                mapping_str = ", ".join([f"{cls} → {code}" for code, cls in enumerate(le.classes_)])
                mapping_list.append([col, mapping_str])
        mapping_df = pd.DataFrame(mapping_list, columns=["Feature", "Encoding"])
        st.dataframe(mapping_df)
