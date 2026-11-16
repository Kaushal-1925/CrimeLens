"""
Crime Detection Using NLP - Interactive Streamlit Frontend
Production Application for Academic Demonstration & Incident Classification
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Guarantee project root is in sys.path for modular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Optional plotting within Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Crime Detection Using NLP",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling - Dark Charcoal Theme with Restrained Accents
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Dark charcoal and neutral surfaces */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header & Titles in Main Content */
    .stMain h1, .stMain h2, .stMain h3, .stMain h4, .stMain h5, .stMain h6,
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Input Field Labels (specifically for text area & inputs) */
    div[data-testid="stTextArea"] label,
    .stTextArea label,
    div[data-testid="stTextInput"] label {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        margin-bottom: 6px !important;
    }

    /* =========================================================
       SIDEBAR THEME & CONTRAST - Guaranteed High Visibility
       ========================================================= */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarNav"] {
        background-color: #141923 !important;
        color: #f1f5f9 !important;
        border-right: 1px solid #232c3d !important;
    }

    /* Sidebar Title & Subtitle */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
    }

    /* Sidebar Radio Header: "Navigation Menu" */
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 8px !important;
    }

    /* Sidebar Radio Option Items (Home, Crime Prediction, etc.) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label div {
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Selected Active Navigation Option */
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #60a5fa !important;
        -webkit-text-fill-color: #60a5fa !important;
        font-weight: 700 !important;
    }

    /* Hover Navigation Option */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover span {
        color: #93c5fd !important;
        -webkit-text-fill-color: #93c5fd !important;
    }

    /* Sidebar Code Badges */
    section[data-testid="stSidebar"] code {
        background-color: #1e2638 !important;
        color: #93c5fd !important;
        -webkit-text-fill-color: #93c5fd !important;
        border: 1px solid #334155 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
    }

    /* Sidebar Horizontal Dividers */
    section[data-testid="stSidebar"] hr {
        border-color: #232c3d !important;
        margin-top: 14px !important;
        margin-bottom: 14px !important;
    }
    
    /* Cards and Surface Panels */
    .surface-card {
        background-color: #1a1f29;
        border: 1px solid #2a3241;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #161b24;
        border: 1px solid #262e3d;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Result Badges */
    .crime-alert-box {
        background-color: rgba(185, 28, 28, 0.15);
        border: 1px solid #dc2626;
        border-left: 6px solid #ef4444;
        border-radius: 6px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    .crime-alert-title {
        color: #fca5a5;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }
    
    .no-crime-alert-box {
        background-color: rgba(22, 101, 52, 0.15);
        border: 1px solid #16a34a;
        border-left: 6px solid #22c55e;
        border-radius: 6px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
    .no-crime-alert-title {
        color: #86efac;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }
    
    /* Legal Disclaimer Callout */
    .disclaimer-card {
        background-color: #182230;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        padding: 14px 18px;
        font-size: 13px;
        color: #cbd5e1;
        margin-top: 15px;
        line-height: 1.5;
    }

    /* Buttons styling */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid #3b82f6;
        font-weight: 500;
        padding: 8px 18px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        border-color: #60a5fa;
    }

    /* Incident Reporting Text Area & Inputs - Pure Black Text on High-Contrast Surface */
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="textarea"] textarea,
    .stTextArea textarea,
    textarea {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        caret-color: #000000 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 6px !important;
    }
    
    div[data-testid="stTextArea"] textarea:focus,
    div[data-baseweb="textarea"] textarea:focus,
    .stTextArea textarea:focus,
    textarea:focus {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: #ffffff !important;
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
    }
    
    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-baseweb="textarea"] textarea::placeholder,
    .stTextArea textarea::placeholder,
    textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        font-weight: 400 !important;
    }

    div[data-baseweb="textarea"] {
        background-color: #ffffff !important;
        border-radius: 6px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Asset Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Loads the serialized trained pipeline from models/crime_classifier.joblib."""
    model_path = ROOT_DIR / "models" / "crime_classifier.joblib"
    if not model_path.exists():
        st.error(f"Trained model not found at: {model_path}. Please execute the Jupyter Notebook first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_metadata():
    """Loads model training metadata and actual test evaluation results."""
    meta_path = ROOT_DIR / "models" / "model_metadata.json"
    if not meta_path.exists():
        st.error(f"Metadata file not found at: {meta_path}.")
        st.stop()
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataset_sample():
    """Loads a sample of the cleaned dataset for the dataset explorer tab."""
    clean_csv_path = ROOT_DIR / "data" / "processed" / "CrimeVsNoCrimeArticles_cleaned.csv"
    if not clean_csv_path.exists():
        clean_csv_path = ROOT_DIR / "data" / "raw" / "CrimeVsNoCrimeArticles.csv"
    if clean_csv_path.exists():
        return pd.read_csv(clean_csv_path)
    return None


# Load Assets
model_pipeline = load_trained_model()
metadata = load_metadata()

# Initialize Session State for Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text_val" not in st.session_state:
    st.session_state.input_text_val = ""


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.markdown("### ⚖️ Crime Detection NLP")
st.sidebar.markdown(
    "<small style='color: #94a3b8;'>Binary Text Classification System<br>Academic ML Project</small>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home",
        "Crime Prediction",
        "Model Performance",
        "Dataset Information",
        "About & Architecture"
    ],
    index=1  # Default to the prediction page for immediate user interaction
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Model:** `{metadata.get('selected_model', 'Multinomial Naive Bayes')}`")
st.sidebar.markdown(f"**Test F1-Score:** `{metadata['test_evaluation_metrics']['Multinomial Naive Bayes']['F1-Score']:.4f}`")
st.sidebar.markdown(f"**Test Accuracy:** `{metadata['test_evaluation_metrics']['Multinomial Naive Bayes']['Accuracy']:.2%}`")
st.sidebar.markdown(
    "<div style='font-size: 11px; color: #64748b; margin-top: 20px;'>"
    "Ground Truth Kaggle Dataset: CrimeVsNoCrimeArticles.csv<br>"
    "Stratified 80:20 Train-Test Split</div>",
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: HOME
# =========================================================
if nav_choice == "Home":
    st.title("Crime Detection Using NLP")
    st.markdown("### Binary Classification of News Headlines & Incident Reports")
    
    st.markdown("""
    <div class="surface-card">
        <h4>Project Overview</h4>
        <p>
            This application is an academic Natural Language Processing (NLP) system designed to triage, analyze,
            and classify natural language incident descriptions and news headlines into binary categories:
        </p>
        <ul>
            <li><strong>Class 1 — CRIME:</strong> Incidents involving violent offenses, robbery, burglary, theft, fraud, homicides, or police investigations.</li>
            <li><strong>Class 0 — NO CRIME:</strong> Everyday civic events, lifestyle journalism, wellness advice, sports updates, and general human interest stories.</li>
        </ul>
        <p>
            Trained and benchmarked on <strong>7,116 verified articles</strong> using an 80:20 stratified holdout protocol,
            the winning <strong>Multinomial Naive Bayes</strong> pipeline demonstrates an <strong>Accuracy of 89.68%</strong>
            and an <strong>F1-Score of 0.8976</strong> on completely unseen test data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Cleaned Corpus</div>
            <div class="metric-value">7,116</div>
            <small style="color: #64748b;">50% Crime / 50% Non-Crime</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value">{metadata['test_evaluation_metrics']['Multinomial Naive Bayes']['Accuracy']:.2%}</div>
            <small style="color: #64748b;">Holdout (1,424 items)</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Recall</div>
            <div class="metric-value">{metadata['test_evaluation_metrics']['Multinomial Naive Bayes']['Recall']:.2%}</div>
            <small style="color: #64748b;">Minimizing Missed Crimes</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test ROC-AUC</div>
            <div class="metric-value">{metadata['test_evaluation_metrics']['Multinomial Naive Bayes']['ROC-AUC']:.4f}</div>
            <small style="color: #64748b;">Threshold Separability</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="surface-card">
        <h4>Methodological Pipeline</h4>
        <ol>
            <li><strong>Data Sanitation:</strong> Audited missing records (1 dropped) and identical duplicate rows (7 dropped).</li>
            <li><strong>Negation-Preserving NLP:</strong> Preprocessed text with custom stopword handling that rigorously preserves negation operators (<code>not</code>, <code>no</code>, <code>never</code>, <code>neither</code>), combined with WordNet lemmatization.</li>
            <li><strong>TF-IDF Vectorization:</strong> Unigram + bigram feature space (top 5,000 features, fitted strictly on training data).</li>
            <li><strong>Model Benchmarking:</strong> Comparative 5-fold cross-validation across Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Machines.</li>
            <li><strong>Model Selection:</strong> Naive Bayes chosen for lowest false negatives (highest recall on actual crimes) and superior F1-score.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚖️ Critical Academic & Legal Notice:</strong><br>
        This tool predicts whether a headline or incident description describes crime-related events based on statistical linguistic patterns in journalistic datasets. It does <strong>not</strong> verify whether a crime legally occurred, establish judicial truth, assess evidentiary guilt, or evaluate the credibility of allegations.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: CRIME PREDICTION (CORE APPLICATION)
# =========================================================
elif nav_choice == "Crime Prediction":
    st.title("Incident Text Classification")
    st.markdown("Enter an incident description, headline, or news excerpt below to obtain an automated classification.")

    # Preset Example Clickers
    st.markdown("##### Quick-Load Representative Examples:")
    sample_cols = st.columns(4)
    
    examples = {
        "🚨 Armed Robbery": "Armed suspect arrested after late night physical assault and store robbery downtown",
        "🔓 Burglary & Theft": "Two masked individuals broke into residential property and stole electronics overnight",
        "🏃 Fitness & Wellness": "What's New and Cool in the Fitness Sphere? Ten easy morning wellness habits",
        "🤝 Community Event": "Local high school students organize annual charity sports festival this weekend",
        "🔍 Subtle / Borderline": "Police department hosts open house coffee session to meet neighborhood residents",
        "🚫 Negation Case": "Court acquits suspect after finding no evidence of burglary or criminal conduct",
        "💼 Business News": "Tech enterprise announces quarterly financial earnings and global expansion strategy",
        "⚖️ Fraud & Scam": "Individual indicted on federal wire fraud and embezzlement charges involving investment fund"
    }

    example_keys = list(examples.keys())
    for i, key in enumerate(example_keys):
        col_idx = i % 4
        with sample_cols[col_idx]:
            if st.button(key, key=f"ex_btn_{i}", use_container_width=True):
                st.session_state.input_text_val = examples[key]

    # Text Input Area
    user_input = st.text_area(
        "Incident Description / Headline Input:",
        value=st.session_state.input_text_val,
        height=130,
        placeholder="Enter an incident description here...",
        help="Paste a news headline, police log summary, or incident description."
    )

    action_col1, action_col2, _ = st.columns([1, 1, 4])
    with action_col1:
        predict_clicked = st.button("Classify Incident", type="primary", use_container_width=True)
    with action_col2:
        clear_clicked = st.button("Clear Input", use_container_width=True)

    if clear_clicked:
        st.session_state.input_text_val = ""
        st.rerun()

    if predict_clicked:
        cleaned_input = user_input.strip()
        if not cleaned_input:
            st.warning("⚠️ Input validation error: Please enter an incident description or select one of the examples above.")
        else:
            with st.spinner("Analyzing text semantics and scoring TF-IDF vectors..."):
                time.sleep(0.15)  # brief visual feedback for user
                
                # Pass directly through loaded production pipeline
                predicted_label = int(model_pipeline.predict([cleaned_input])[0])
                predicted_probs = model_pipeline.predict_proba([cleaned_input])[0]
                
                crime_prob = float(predicted_probs[1])
                non_crime_prob = float(predicted_probs[0])
                confidence = float(predicted_probs[predicted_label]) * 100
                model_name = metadata.get("selected_model", "Multinomial Naive Bayes")
                
                # Record to Session History
                st.session_state.history.insert(0, {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Input Text": cleaned_input[:80] + ("..." if len(cleaned_input) > 80 else ""),
                    "Classification": "CRIME (1)" if predicted_label == 1 else "NO CRIME (0)",
                    "Model Confidence": f"{confidence:.1f}%",
                    "P(Crime)": f"{crime_prob:.3f}"
                })

            # Display Classification Output
            if predicted_label == 1:
                st.markdown(f"""
                <div class="crime-alert-box">
                    <div class="crime-alert-title">🚨 CRIME-RELATED INCIDENT DETECTED</div>
                    <div style="font-size: 15px; color: #f1f5f9; margin-bottom: 8px;">
                        The model classified this text as <strong>describing a crime-related incident</strong>.
                    </div>
                    <div style="font-size: 13px; color: #cbd5e1;">
                        <strong>Model:</strong> {model_name} &nbsp;|&nbsp; 
                        <strong>Model Confidence:</strong> {confidence:.2f}% &nbsp;|&nbsp; 
                        <strong>P(Crime):</strong> {crime_prob:.4f} &nbsp;|&nbsp; 
                        <strong>P(Non-Crime):</strong> {non_crime_prob:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="no-crime-alert-box">
                    <div class="no-crime-alert-title">✅ NO CRIME-RELATED INCIDENT DETECTED</div>
                    <div style="font-size: 15px; color: #f1f5f9; margin-bottom: 8px;">
                        The model classified this text as <strong>non-crime-related</strong>.
                    </div>
                    <div style="font-size: 13px; color: #cbd5e1;">
                        <strong>Model:</strong> {model_name} &nbsp;|&nbsp; 
                        <strong>Model Confidence:</strong> {confidence:.2f}% &nbsp;|&nbsp; 
                        <strong>P(Non-Crime):</strong> {non_crime_prob:.4f} &nbsp;|&nbsp; 
                        <strong>P(Crime):</strong> {crime_prob:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability Breakdown Progress Bars
            st.markdown("##### Model Decision Probability Breakdown:")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.write(f"**P(Crime-Related):** `{crime_prob:.2%}`")
                st.progress(crime_prob)
            with prob_col2:
                st.write(f"**P(Non-Crime / Benign):** `{non_crime_prob:.2%}`")
                st.progress(non_crime_prob)

            # Ethical/Academic Clarification
            st.markdown("""
            <div class="disclaimer-card">
                <strong>💡 Interpretability Note:</strong><br>
                Model confidence represents the statistical likelihood under the TF-IDF feature distribution that this text matches linguistic patterns found in crime reporting. It is <strong>not</strong> the probability that an actual real-world crime occurred.
            </div>
            """, unsafe_allow_html=True)

    # Display Current Session History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Session Classification History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================
elif nav_choice == "Model Performance":
    st.title("Model Benchmarking & Evaluation")
    st.markdown("Comparative performance metrics computed on the **untouched 20% holdout test set (1,424 samples)**.")

    metrics_data = metadata.get("test_evaluation_metrics", {})
    
    # Summary Table
    rows = []
    for model_name, m in metrics_data.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{m['Accuracy']:.2%}",
            "Precision": f"{m['Precision']:.2%}",
            "Recall": f"{m['Recall']:.2%}",
            "F1-Score": f"{m['F1-Score']:.4f}",
            "ROC-AUC": f"{m['ROC-AUC']:.4f}",
            "Missed Crimes (FN)": int(m['False Negatives']),
            "False Alarms (FP)": int(m['False Positives'])
        })
    summary_df = pd.DataFrame(rows).set_index("Model")
    
    st.markdown("#### Test Set Performance Summary:")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")
    
    # Detailed Metric Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Performance Metric Comparison")
        # Visual grouped bar chart
        plot_records = []
        for model_name, m in metrics_data.items():
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
                plot_records.append({'Model': model_name, 'Metric': metric, 'Score': m[metric]})
        p_df = pd.DataFrame(plot_records)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=p_df, x='Metric', y='Score', hue='Model', 
                    palette=['#3b82f6', '#ef4444', '#10b981'], ax=ax, edgecolor='black')
        ax.set_ylim(0.80, 1.0)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title("Test Set Evaluation Metrics", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f"#### Confusion Matrix ({metadata.get('selected_model')})")
        mnb_m = metrics_data.get(metadata.get('selected_model', 'Multinomial Naive Bayes'))
        cm = np.array([
            [int(mnb_m['True Negatives']), int(mnb_m['False Positives'])],
            [int(mnb_m['False Negatives']), int(mnb_m['True Positives'])]
        ])
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred No Crime', 'Pred Crime'],
            yticklabels=['Actual No Crime', 'Actual Crime'],
            annot_kws={'size': 14, 'weight': 'bold'},
            ax=ax_cm
        )
        ax_cm.set_title(f"Test Confusion Matrix (N = {cm.sum():,})", fontsize=12, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()

    st.markdown("""
    <div class="surface-card">
        <h4>Why Multinomial Naive Bayes was Selected:</h4>
        <ol>
            <li><strong>Highest Recall ({recall:.2%}):</strong> In incident detection, a False Negative (missing a real violent crime) is significantly more detrimental than a False Positive. MNB missed only <strong>{fn} crimes</strong> out of 712 actual crime cases (error rate of 8.99%), outperforming Linear SVM (84 missed) and Logistic Regression (103 missed).</li>
            <li><strong>Highest F1-Score ({f1:.4f}):</strong> Balances precision and recall optimally.</li>
            <li><strong>Highest ROC-AUC ({auc:.4f}):</strong> Demonstrates the highest global threshold separability across varying decision boundaries.</li>
            <li><strong>Zero Computational Overhead:</strong> Ultra-fast inference (< 5 ms latency), zero GPU dependency, and exceptional scalability for production servers.</li>
        </ol>
    </div>
    """.format(
        recall=mnb_m['Recall'],
        fn=int(mnb_m['False Negatives']),
        f1=mnb_m['F1-Score'],
        auc=mnb_m['ROC-AUC']
    ), unsafe_allow_html=True)


# =========================================================
# PAGE 4: DATASET INFORMATION
# =========================================================
elif nav_choice == "Dataset Information":
    st.title("Dataset Information & Quality Audit")
    st.markdown("Ground-truth data audited and verified from Kaggle: `CrimeVsNoCrimeArticles.csv`.")

    stats = metadata.get("dataset_statistics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Raw Total Records</div>
            <div class="metric-value">{stats.get('total_raw_records', 7124):,}</div>
            <small style="color: #64748b;">Original Kaggle archive</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing Titles Removed</div>
            <div class="metric-value">{stats.get('missing_titles_removed', 1)}</div>
            <small style="color: #64748b;">Row index 1353</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duplicates Removed</div>
            <div class="metric-value">{stats.get('duplicates_removed', 7)}</div>
            <small style="color: #64748b;">Exact headline matches</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Final Cleaned Records</div>
            <div class="metric-value">{stats.get('total_cleaned_records', 7116):,}</div>
            <small style="color: #64748b;">5,692 train / 1,424 test</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="surface-card">
        <h4>Dataset Structure & Schema</h4>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="border-bottom: 1px solid #334155; text-align: left; color: #94a3b8;">
                    <th style="padding: 8px;">Column Name</th>
                    <th style="padding: 8px;">Role</th>
                    <th style="padding: 8px;">Data Type</th>
                    <th style="padding: 8px;">Description</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px;"><code>title</code></td>
                    <td style="padding: 8px;">Input Feature ($X$)</td>
                    <td style="padding: 8px;">String (Text)</td>
                    <td style="padding: 8px;">Article headline or incident summary</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><code>is_crime_report</code></td>
                    <td style="padding: 8px;">Target Label ($y$)</td>
                    <td style="padding: 8px;">Integer (Binary 0 / 1)</td>
                    <td style="padding: 8px;">1 = Crime Incident, 0 = Non-Crime Incident</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Sample Data View
    sample_data = load_dataset_sample()
    if sample_data is not None:
        st.markdown("#### Sample Records from Cleaned Dataset:")
        st.dataframe(sample_data[['title', 'is_crime_report']].head(15), use_container_width=True)


# =========================================================
# PAGE 5: ABOUT & ARCHITECTURE
# =========================================================
elif nav_choice == "About & Architecture":
    st.title("About the System Architecture")
    st.markdown("Technical documentation, natural language processing design, and algorithmic considerations.")

    st.markdown("""
    <div class="surface-card">
        <h4>Natural Language Processing (NLP) Design</h4>
        <p>
            Standard text preprocessing often blindly discards stopwords. In crime classification, this causes fatal semantic inversions.
            For instance:
        </p>
        <ul>
            <li><code>"Court finds no evidence of assault"</code> $\\to$ blind removal turns this into <code>"Court finds evidence assault"</code>, converting an acquittal into a crime prediction.</li>
            <li>Our pipeline utilizes a <strong>negation-preserving stopword policy</strong> where words like <code>no</code>, <code>not</code>, <code>never</code>, <code>neither</code>, <code>nor</code>, <code>without</code>, and <code>against</code> are preserved.</li>
        </ul>
        <p>
            WordNet lemmatization normalizes morphological variants (e.g., <em>robbing, robbed, robberies</em>) into canonical root lemmas while filtering single-character noise.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="surface-card">
        <h4>TF-IDF Feature Space</h4>
        <p>
            Text documents are converted into numerical vector spaces using Term Frequency-Inverse Document Frequency (TF-IDF):
        </p>
        <ul>
            <li><strong>N-Gram Range:</strong> (1, 2) — Captures individual words and compound phrases (e.g., <code>armed robbery</code>, <code>not guilty</code>).</li>
            <li><strong>Max Features:</strong> 5,000 top n-grams by frequency and variance.</li>
            <li><strong>Min Document Frequency:</strong> 2 occurrences minimum to prune typographical errors.</li>
            <li><strong>Sublinear Scaling & L2 Norm:</strong> Normalizes document vectors to unit hypersphere for Euclidean invariance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="surface-card">
        <h4>Model Comparison Overview</h4>
        <ul>
            <li><strong>Multinomial Naive Bayes (Winner):</strong> Best-in-class F1-score and highest recall. Particularly resilient to sparse text matrices.</li>
            <li><strong>Linear SVM (Calibrated):</strong> Excellent margin separation with Platt-calibrated sigmoid probabilities.</li>
            <li><strong>Logistic Regression:</strong> Strong baseline with L2 penalty, slightly more prone to false negatives on nuanced headlines.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚖️ Ethical & Operational Limitations:</strong><br>
        1. <strong>Headline Bias & Clickbait:</strong> News headlines are drafted to maximize engagement and can exhibit hyperbole or sensational phrasing.<br>
        2. <strong>Metaphorical Language:</strong> Sports headlines (e.g., <em>"Team slaughtered in championship"</em>) or economic headlines (e.g., <em>"Stock crash kills savings"</em>) employ violent lexicons in non-criminal contexts.<br>
        3. <strong>Classification vs Verification:</strong> The model classifies text syntax and vocabulary; it does not substantiate legal truth or replace human judicial scrutiny.
    </div>
    """, unsafe_allow_html=True)

