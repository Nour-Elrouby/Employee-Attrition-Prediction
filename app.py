import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="◈",
    layout="centered",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #f8fafc;
            --muted: #94a3b8;
            --brand: #5b7cfa;
            --brand-dark: #4667e8;
            --surface: #111827;
            --canvas: #080b12;
            --line: #253044;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
            font-family: "Aptos", "Segoe UI", Arial, sans-serif;
        }

        [data-testid="stHeader"] {
            background: rgba(8, 11, 18, 0.92);
        }

        .block-container {
            max-width: 900px;
            padding-top: 3.5rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, label,
        [data-testid="stMarkdownContainer"] p {
            font-family: "Aptos", "Segoe UI", Arial, sans-serif;
        }

        .eyebrow {
            color: var(--brand);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 750;
            letter-spacing: -0.04em;
            line-height: 1.08;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.65;
            margin: 0.8rem 0 2rem;
            max-width: 650px;
        }

        [data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
            padding: 1.5rem 1.5rem 1.2rem;
        }

        [data-testid="stWidgetLabel"] p {
            color: #dbe4f0;
            font-size: 0.9rem;
            font-weight: 650;
        }

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
            background: #0b1220;
            border-color: #334155;
            color: var(--ink);
        }

        [data-baseweb="select"] *,
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] input {
            color: var(--ink) !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="menu"] {
            background: #111827 !important;
            color: var(--ink) !important;
        }

        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: var(--brand);
            border-color: var(--brand);
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: var(--brand);
            border: 1px solid var(--brand);
            border-radius: 10px;
            color: #ffffff;
            font-weight: 700;
            min-height: 3rem;
            transition: all 160ms ease;
            width: 100%;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: var(--brand-dark);
            border-color: var(--brand-dark);
            box-shadow: 0 5px 18px rgba(91, 124, 250, 0.3);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .result-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-left: 5px solid var(--accent);
            border-radius: 14px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
            margin: 1.5rem 0 1rem;
            padding: 1.3rem 1.5rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-value {
            color: var(--ink);
            font-size: 2.25rem;
            font-weight: 750;
            letter-spacing: -0.035em;
            line-height: 1.15;
            margin: 0.3rem 0;
        }

        .result-message {
            color: #cbd5e1;
            font-size: 0.95rem;
        }

        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        hr {
            border-color: var(--line) !important;
            margin-top: 2.5rem !important;
        }

        [data-testid="stCaptionContainer"] p {
            color: #8491a5;
        }

        [data-testid="stImage"] img,
        [data-testid="stPlotlyChart"],
        [data-testid="stPyplot"] {
            border-radius: 14px;
            overflow: hidden;
        }

        @media (max-width: 640px) {
            .block-container { padding-top: 2rem; }
            [data-testid="stForm"] { padding: 1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

#load trained model and reference columns
@st.cache_resource
def load_model():
    import pickle
    with open('lgbm_model.pkl','rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl','rb') as f:
        columns = pickle.load(f)
    return model,columns

model,model_columns = load_model()

st.markdown(
    """
    <div class="eyebrow">People analytics</div>
    <h1 class="hero-title">Employee attrition predictor</h1>
    <p class="hero-copy">
        Estimate an employee's attrition risk and understand the factors shaping
        the prediction.
    </p>
    """,
    unsafe_allow_html=True,
)

#input form - top SHAP drivers only, not all 30+ features
with st.form("attrition_form"):
    col1,col2 = st.columns(2)

    with col1:
        overtime = st.selectbox("OverTime",["Yes","No"])
        job_satisfaction = st.slider("Job Satisfaction (1-4)",1,4,3)
        monthly_income = st.number_input("Monthly Income ($)",1000,20000,5000,step=100)
        years_at_company = st.number_input("Years at Company",0,40,3)

    with col2:
        env_satisfaction = st.slider("Environment Satisfaction (1-4)",1,4,3)
        work_life_balance = st.slider("Work Life Balance (1-4)",1,4,3)
        years_since_promotion = st.number_input("Years Since Last Promotion",0,15,1)
        job_level = st.selectbox("Job Level",[1,2,3,4,5])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    #build a single-row input matching training feature engineering
    satisfaction_index = (job_satisfaction + env_satisfaction + work_life_balance) / 3
    income_to_level_ratio = monthly_income / job_level
    promotion_gap = years_at_company - years_since_promotion

    input_dict = {
        "OverTime_Yes"          : 1 if overtime == "Yes" else 0,
        "JobSatisfaction"       : job_satisfaction,
        "EnvironmentSatisfaction": env_satisfaction,
        "WorkLifeBalance"       : work_life_balance,
        "MonthlyIncome"         : monthly_income,
        "YearsAtCompany"        : years_at_company,
        "YearsSinceLastPromotion": years_since_promotion,
        "JobLevel"              : job_level,
        "SatisfactionIndex"     : satisfaction_index,
        "IncomeToLevelRatio"    : income_to_level_ratio,
        "PromotionGap"          : promotion_gap,
    }

    #align with training columns, fill missing with 0
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns,fill_value=0)

    #predict
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_pct = round(risk_prob * 100,1)

    if risk_prob >= 0.5:
        risk_label = "High risk"
        risk_message = "Recommend scheduling a focused retention conversation."
        risk_color = "#d92d20"
    elif risk_prob >= 0.3:
        risk_label = "Moderate risk"
        risk_message = "Monitor the situation and review the employee experience."
        risk_color = "#dc6803"
    else:
        risk_label = "Low risk"
        risk_message = "Current indicators suggest a comparatively low attrition risk."
        risk_color = "#5b7cfa"

    st.markdown(
        f"""
        <div class="result-card" style="--accent: {risk_color};">
            <div class="result-label">Predicted attrition risk</div>
            <div class="result-value">{risk_pct}% · {risk_label}</div>
            <div class="result-message">{risk_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SHAP explanation for this specific prediction
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_row = np.asarray(shap_values[-1][0])
        else:
            shap_row = np.asarray(shap_values[0])

        if shap_row.size == len(model_columns) and np.isfinite(shap_row).all():
            st.subheader("What is driving this prediction?")
            top_indices = np.argsort(np.abs(shap_row))[-8:]
            contributions = shap_row[top_indices]
            feature_names = [
                str(model_columns[i]).replace("_", " ") for i in top_indices
            ]
            bar_colors = ["#f4b740" if value > 0 else "#5b7cfa" for value in contributions]

            with plt.style.context("dark_background"):
                fig, ax = plt.subplots(figsize=(9, 4.8))
                fig.patch.set_facecolor("#080b12")
                ax.set_facecolor("#080b12")
                ax.barh(feature_names, contributions, color=bar_colors, height=0.58)
                ax.axvline(0, color="#64748b", linewidth=1)
                ax.set_xlabel("Impact on attrition risk", color="#94a3b8", labelpad=10)
                ax.tick_params(colors="#cbd5e1", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(axis="x", color="#334155", alpha=0.55, linewidth=0.7)
                ax.set_axisbelow(True)
                fig.tight_layout(pad=1.5)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            st.caption("Amber raises predicted risk; sapphire lowers it.")
    except Exception:
        # Never display a broken or empty explanation panel.
        pass

st.divider()
st.caption("LightGBM model · IBM HR Analytics Employee Attrition dataset")
