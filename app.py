import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Epileptic Seizure Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .seizure-box {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .normal-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model_path = 'model/seizure_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path), True
        return None, False
    except Exception:
        return None, False

@st.cache_data
def load_sample_data():
    path = 'data/Epileptic Seizure Recognition.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def preprocess_input(X_raw):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

def plot_eeg_signal(signal, title, color='#e74c3c', prediction=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(signal, color=color, linewidth=1.2)
    ax.fill_between(range(len(signal)), signal, alpha=0.15, color=color)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Time Steps'); ax.set_ylabel('Amplitude')
    ax.grid(alpha=0.3); ax.set_xlim([0, len(signal)])
    if prediction is not None:
        label = "SEIZURE DETECTED" if prediction == 1 else "NO SEIZURE"
        box_color = '#e74c3c' if prediction == 1 else '#2ecc71'
        ax.text(0.98, 0.92, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='white',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, alpha=0.9))
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.markdown("## 🧠 Seizure Detection")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "🔍 Predict", "📊 EDA Visualizations", "ℹ️ Model Info"])
    st.markdown("---")
    model, model_loaded = load_model()
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Model not found\nTrain the model first.")
    st.markdown("---")
    st.markdown("**Dataset:** UCI Epileptic Seizure")
    st.markdown("**Samples:** 11,500")
    st.markdown("**Features:** 178 EEG time steps")
    st.markdown("**Task:** Binary Classification")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">🧠 Epileptic Seizure Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning-based EEG Signal Classification</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>11,500</h2><p>EEG Samples</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>178</h2><p>EEG Features</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>3</h2><p>DL Models</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>92%</h2><p>Target Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 About the Project")
        st.markdown("""
        This project uses **deep learning models** to automatically detect epileptic seizures
        from EEG (Electroencephalogram) signals.

        Epilepsy affects over **50 million people** worldwide. Early and accurate seizure
        detection can significantly improve patient outcomes and reduce false alarms.

        **What this app does:**
        - Upload EEG signal data (CSV format)
        - Automatically classifies as **Seizure** or **Non-Seizure**
        - Visualizes the EEG signal with the prediction
        - Shows confidence score
        """)

    with col2:
        st.markdown("### 🤖 Models Used")
        st.markdown("""
        | Model | Description |
        |---|---|
        | **1D CNN** | Captures local EEG patterns using convolution |
        | **LSTM** | Learns temporal dependencies in EEG sequences |
        | **CNN-LSTM** | Combines spatial + temporal feature extraction |

        ### 📊 Dataset
        **UCI Epileptic Seizure Recognition**
        - 5 classes: 1 seizure + 4 non-seizure types
        - Perfectly balanced: 2,300 samples each
        - Binary task: Seizure (Class 1) vs Non-Seizure (2–5)
        """)

    st.markdown("---")
    st.markdown("### 🔬 Class Description")
    col1, col2, col3, col4, col5 = st.columns(5)
    classes = [
        ("🔴", "Class 1", "Seizure", "#ffe6e6"),
        ("🔵", "Class 2", "Tumour Area", "#e6f0ff"),
        ("🟢", "Class 3", "Healthy Area", "#e6fff0"),
        ("🟡", "Class 4", "Eyes Closed", "#fffce6"),
        ("🟣", "Class 5", "Eyes Open", "#f5e6ff"),
    ]
    for col, (emoji, cls, desc, bg) in zip([col1, col2, col3, col4, col5], classes):
        with col:
            st.markdown(f"""
            <div style='background:{bg};padding:1rem;border-radius:10px;text-align:center;'>
            <h3>{emoji}</h3><b>{cls}</b><br><small>{desc}</small>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown("## 🔍 EEG Seizure Prediction")

    tab1, tab2 = st.tabs(["📁 Upload CSV", "🎲 Try Sample Data"])

    with tab1:
        st.markdown("### Upload EEG Data")
        st.markdown('<div class="info-box">Upload a CSV file with 178 EEG feature columns (same format as the dataset). Each row = one EEG sample.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                st.success(f"✅ File uploaded: {df_upload.shape[0]} samples, {df_upload.shape[1]} columns")

                feature_cols = [c for c in df_upload.columns if c.startswith('X')]
                if len(feature_cols) == 178:
                    X_input = df_upload[feature_cols].values
                elif df_upload.shape[1] == 178:
                    X_input = df_upload.values
                else:
                    st.error(f"Expected 178 EEG feature columns, got {df_upload.shape[1]}. Please check your file format.")
                    X_input = None

                if X_input is not None:
                    if not model_loaded:
                        st.warning("⚠️ Model not trained yet. Train the model first by running `seizure_detection.ipynb`.")
                    else:
                        X_proc = preprocess_input(X_input)
                        with st.spinner("Running prediction..."):
                            probs = model.predict(X_proc, verbose=0).ravel()
                            preds = (probs >= 0.5).astype(int)

                        n_seizure = preds.sum()
                        n_normal  = len(preds) - n_seizure

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Samples", len(preds))
                        col2.metric("🔴 Seizure Detected", int(n_seizure))
                        col3.metric("🟢 Normal", int(n_normal))

                        result_df = pd.DataFrame({
                            'Sample': range(1, len(preds)+1),
                            'Prediction': ['🔴 SEIZURE' if p == 1 else '🟢 NORMAL' for p in preds],
                            'Confidence': [f"{max(p, 1-p)*100:.1f}%" for p in probs]
                        })
                        st.dataframe(result_df, use_container_width=True)

                        st.markdown("### EEG Signals")
                        n_show = min(3, len(X_input))
                        for i in range(n_show):
                            color = '#e74c3c' if preds[i] == 1 else '#2ecc71'
                            fig = plot_eeg_signal(X_input[i], f"Sample {i+1}", color, preds[i])
                            st.pyplot(fig); plt.close()

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab2:
        st.markdown("### Try with Sample Data from Dataset")

        df_sample = load_sample_data()
        if df_sample is None:
            st.warning("Dataset not found in `data/` folder.")
        else:
            feature_cols = [c for c in df_sample.columns if c.startswith('X')]
            col1, col2 = st.columns(2)
            with col1:
                class_filter = st.selectbox("Pick a class to sample from:", {
                    1: "Class 1 — Seizure",
                    2: "Class 2 — Tumour Area",
                    3: "Class 3 — Healthy Area",
                    4: "Class 4 — Eyes Closed",
                    5: "Class 5 — Eyes Open"
                })
            with col2:
                n_samples = st.slider("Number of samples", 1, 10, 3)

            if st.button("🎲 Run Prediction on Sample"):
                subset = df_sample[df_sample['y'] == class_filter].sample(n_samples, random_state=42)
                X_input = subset[feature_cols].values
                true_labels = subset['y'].values

                if not model_loaded:
                    st.warning("⚠️ Model not trained yet. Showing EEG signals only.")
                    for i in range(len(X_input)):
                        true_cls = true_labels[i]
                        color = '#e74c3c' if true_cls == 1 else '#3498db'
                        title = f"Sample {i+1} | True Class: {true_cls} ({'Seizure' if true_cls==1 else 'Non-Seizure'})"
                        fig = plot_eeg_signal(X_input[i], title, color)
                        st.pyplot(fig); plt.close()
                else:
                    X_proc = preprocess_input(X_input)
                    probs = model.predict(X_proc, verbose=0).ravel()
                    preds = (probs >= 0.5).astype(int)

                    for i in range(len(X_input)):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            color = '#e74c3c' if preds[i] == 1 else '#2ecc71'
                            fig = plot_eeg_signal(X_input[i], f"Sample {i+1}", color, preds[i])
                            st.pyplot(fig); plt.close()
                        with col_b:
                            st.markdown("**Prediction:**")
                            if preds[i] == 1:
                                st.markdown('<div class="seizure-box">🔴 SEIZURE</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="normal-box">🟢 NORMAL</div>', unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {max(probs[i], 1-probs[i])*100:.1f}%")
                            true_label = "Seizure" if true_labels[i] == 1 else "Non-Seizure"
                            correct = (preds[i] == 1) == (true_labels[i] == 1)
                            st.markdown(f"**True Label:** {true_label}")
                            st.markdown("✅ Correct" if correct else "❌ Incorrect")

# ══════════════════════════════════════════════════════════════════════════════
# EDA VISUALIZATIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Visualizations":
    st.markdown("## 📊 Exploratory Data Analysis")

    plots = {
        "Class Distribution": "outputs/eda_class_distribution.png",
        "EEG Signals Per Class": "outputs/eda_eeg_signals.png",
        "Average Signal Shape Per Class": "outputs/eda_avg_signal_per_class.png",
        "PCA Visualization": "outputs/eda_pca.png",
        "Signal Statistics": "outputs/eda_signal_stats.png",
        "Amplitude Distribution": "outputs/eda_amplitude_distribution.png",
        "Feature Variance": "outputs/eda_feature_variance.png",
        "Correlation Heatmap": "outputs/eda_correlation_heatmap.png",
    }

    keys = list(plots.keys())
    for i in range(0, len(keys), 2):
        col1, col2 = st.columns(2)
        for col, key in zip([col1, col2], keys[i:i+2]):
            with col:
                path = plots[key]
                if os.path.exists(path):
                    st.markdown(f"**{key}**")
                    st.image(path, use_container_width=True)
                else:
                    st.info(f"{key} — Run `eda.ipynb` to generate this plot.")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":
    st.markdown("## ℹ️ Model Architecture Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔴 1D CNN")
        st.code("""Input (178, 1)
Conv1D(64, k=5) + BN
MaxPool + Dropout(0.3)

Conv1D(128, k=5) + BN
MaxPool + Dropout(0.3)

Conv1D(256, k=3) + BN
MaxPool + Dropout(0.3)

Flatten
Dense(128) + Dropout(0.5)
Dense(64)  + Dropout(0.3)
Dense(1, sigmoid)""", language="text")
        st.markdown("**Strength:** Fast, captures local EEG patterns")

    with col2:
        st.markdown("### 🔵 LSTM")
        st.code("""Input (178, 1)

LSTM(128, return_seq=True)
Dropout(0.3)

LSTM(64)
Dropout(0.3)

Dense(64)
Dropout(0.3)
Dense(1, sigmoid)""", language="text")
        st.markdown("**Strength:** Captures long-range temporal dependencies")

    with col3:
        st.markdown("### 🟢 CNN-LSTM Hybrid")
        st.code("""Input (178, 1)
Conv1D(64, k=5) + BN
MaxPool + Dropout(0.3)

Conv1D(128, k=3) + BN
MaxPool + Dropout(0.3)

LSTM(128, return_seq=True)
Dropout(0.3)
LSTM(64)
Dropout(0.3)

Dense(64) + Dropout(0.3)
Dense(1, sigmoid)""", language="text")
        st.markdown("**Strength:** Best of both — spatial + temporal")

    st.markdown("---")
    st.markdown("### ⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Loss | Binary Crossentropy |
        | Optimizer | Adam (lr=0.001) |
        | Batch Size | 128 |
        | Max Epochs | 30 |
        | Val Split | 15% |
        """)
    with col2:
        st.markdown("""
        | Callback | Setting |
        |---|---|
        | EarlyStopping | patience=8, restore best |
        | ReduceLROnPlateau | factor=0.5, patience=4 |
        | ModelCheckpoint | save best only |
        """)

    st.markdown("---")
    st.markdown("### 📦 Tech Stack")
    cols = st.columns(6)
    techs = ["Python", "TensorFlow", "Keras", "scikit-learn", "Pandas", "Streamlit"]
    for col, tech in zip(cols, techs):
        col.markdown(f"**{tech}**")

    if model_loaded:
        st.markdown("---")
        st.markdown("### 🏆 Loaded Model Summary")
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        st.code(f.getvalue(), language="text")
