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
    page_title="Seizure Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0d0d0d;
        color: #e0e0e0;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e74c3c33;
        border-radius: 18px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.1rem;
        color: #8899aa;
        margin: 0;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        flex: 1;
        background: #161616;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.3rem 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #e74c3c55; }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 800;
        color: #e74c3c;
        line-height: 1;
    }
    .metric-card .lbl {
        font-size: 0.8rem;
        color: #667;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Result boxes */
    .result-seizure {
        background: linear-gradient(135deg, #ff416c22, #ff4b2b22);
        border: 2px solid #e74c3c;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-seizure .label { font-size: 1.6rem; font-weight: 800; color: #e74c3c; }

    .result-normal {
        background: linear-gradient(135deg, #11998e22, #38ef7d22);
        border: 2px solid #2ecc71;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-normal .label { font-size: 1.6rem; font-weight: 800; color: #2ecc71; }

    .conf { font-size: 0.95rem; color: #aaa; margin-top: 0.4rem; }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        border-left: 4px solid #e74c3c;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Info box */
    .info-box {
        background: #161616;
        border: 1px solid #2a2a2a;
        border-left: 4px solid #3498db;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #aab;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Class badge */
    .badge {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
    .badge .emoji { font-size: 1.8rem; }
    .badge .cls { font-weight: 700; color: #e0e0e0; font-size: 0.95rem; }
    .badge .desc { color: #667; font-size: 0.8rem; margin-top: 0.2rem; }

    /* Model arch card */
    .arch-card {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
    }
    .arch-card h4 { color: #e74c3c; font-size: 1rem; font-weight: 700; margin-bottom: 1rem; }

    /* Sidebar nav */
    div[data-testid="stRadio"] > label { color: #aaa !important; font-size: 0.85rem; }
    div[data-testid="stRadio"] > div > label { color: #e0e0e0 !important; }

    /* Status pill */
    .pill-ok {
        display: inline-block;
        background: #1a3a2a;
        border: 1px solid #2ecc71;
        color: #2ecc71;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .pill-warn {
        display: inline-block;
        background: #3a2a1a;
        border: 1px solid #f39c12;
        color: #f39c12;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #111; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #888; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #1a1a1a !important; color: #e74c3c !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        font-size: 0.95rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Slider, selectbox */
    .stSelectbox > div > div { background: #161616 !important; border-color: #333 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import onnxruntime as ort
        path = 'model/seizure_model.onnx'
        if os.path.exists(path):
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            return session, True
        return None, False
    except Exception:
        return None, False


def _predict(session, X_proc):
    inp_name = session.get_inputs()[0].name
    return session.run(None, {inp_name: X_proc.astype(np.float32)})[0].ravel()

@st.cache_data
def load_dataset():
    path = 'data/Epileptic Seizure Recognition.csv'
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_resource
def get_scaler():
    from sklearn.preprocessing import StandardScaler
    df = load_dataset()
    if df is not None:
        feature_cols = [c for c in df.columns if c.startswith('X')]
        sc = StandardScaler()
        sc.fit(df[feature_cols].values)
        return sc
    return None

def preprocess(X_raw):
    sc = get_scaler()
    if sc is not None:
        X_sc = sc.transform(X_raw)
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_raw)
    return X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)

def plot_signal(signal, title, color, pred=None):
    fig, ax = plt.subplots(figsize=(12, 2.8))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#0d0d0d')
    ax.plot(signal, color=color, linewidth=1.1, alpha=0.95)
    ax.fill_between(range(len(signal)), signal, alpha=0.12, color=color)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#e0e0e0', pad=10)
    ax.set_xlabel('Time Steps', color='#555', fontsize=9)
    ax.set_ylabel('Amplitude', color='#555', fontsize=9)
    ax.tick_params(colors='#444')
    ax.spines[:].set_color('#222')
    ax.grid(alpha=0.12, color='#333')
    ax.set_xlim([0, len(signal)])
    if pred is not None:
        label = "⚠ SEIZURE" if pred == 1 else "✓ NORMAL"
        bc = '#e74c3c' if pred == 1 else '#2ecc71'
        ax.text(0.99, 0.91, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=bc,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#111', edgecolor=bc, linewidth=1.5))
    plt.tight_layout()
    return fig


# ── Load resources ─────────────────────────────────────────────────────────────
model, model_loaded = load_model()
df_data = load_dataset()

CLASS_NAMES = {1:'Seizure', 2:'Tumour Area', 3:'Healthy Area', 4:'Eyes Closed', 5:'Eyes Open'}
CLASS_EMOJIS = {1:'🔴', 2:'🔵', 3:'🟢', 4:'🟡', 5:'🟣'}
CLASS_COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:3rem;'>🧠</div>
        <div style='font-size:1.1rem;font-weight:800;color:#e0e0e0;'>Seizure Detection</div>
        <div style='font-size:0.75rem;color:#555;margin-top:0.2rem;'>EEG · Deep Learning · AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#222;margin:1rem 0;'>", unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠  Home", "🔍  Predict", "📊  EDA Plots", "ℹ️  Model Info"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#222;margin:1rem 0;'>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown('<span class="pill-ok">● Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-warn">● Model Not Trained</span>', unsafe_allow_html=True)
        st.markdown("<div style='color:#555;font-size:0.78rem;margin-top:0.5rem;'>Run seizure_detection.ipynb to train.</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#222;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#444;font-size:0.78rem;'>Built by <b style='color:#666;'>Sandhya Singh</b></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="hero">
        <h1>🧠 Epileptic Seizure Detection</h1>
        <p>AI-powered EEG signal classification using 1D CNN · LSTM · CNN-LSTM Hybrid</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card"><div class="val">11,500</div><div class="lbl">EEG Samples</div></div>
        <div class="metric-card"><div class="val">178</div><div class="lbl">Time Steps</div></div>
        <div class="metric-card"><div class="val">3</div><div class="lbl">DL Models</div></div>
        <div class="metric-card"><div class="val">92%</div><div class="lbl">Target Accuracy</div></div>
        <div class="metric-card"><div class="val">-30%</div><div class="lbl">False Alarms</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#aaa;line-height:1.8;font-size:0.95rem;'>
        Epilepsy affects over <b style='color:#e0e0e0;'>50 million people</b> worldwide.
        This system automatically detects epileptic seizures from raw EEG signals
        using deep learning — enabling faster diagnosis and reducing missed detections.<br><br>
        Upload any EEG CSV file and get an instant prediction with confidence score
        and signal visualization.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Dataset Classes</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, (col, cls) in enumerate(zip(cols, range(1, 6))):
            with col:
                st.markdown(f"""
                <div class="badge">
                    <div class="emoji">{CLASS_EMOJIS[cls]}</div>
                    <div class="cls">Class {cls}</div>
                    <div class="desc">{CLASS_NAMES[cls]}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("01", "Upload", "CSV file with 178 EEG features", "#e74c3c"),
        ("02", "Preprocess", "Normalize & reshape signal", "#3498db"),
        ("03", "Predict", "Deep learning model inference", "#9b59b6"),
        ("04", "Result", "Seizure / Normal + confidence", "#2ecc71"),
    ]
    for col, (num, title, desc, color) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div style='background:#111;border:1px solid #222;border-top:3px solid {color};
                        border-radius:12px;padding:1.2rem;text-align:center;'>
                <div style='font-size:1.8rem;font-weight:800;color:{color};opacity:0.4;'>{num}</div>
                <div style='font-weight:700;color:#e0e0e0;margin:0.3rem 0;'>{title}</div>
                <div style='color:#555;font-size:0.8rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Predict":
    st.markdown('<div class="section-header">EEG Seizure Prediction</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁  Upload CSV", "🎲  Sample from Dataset"])

    with tab1:
        st.markdown('<div class="info-box">Upload a CSV with 178 EEG columns (X1–X178). Each row = one EEG recording.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=['csv'], label_visibility="collapsed")

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                feature_cols = [c for c in df_up.columns if c.startswith('X')]
                X_input = df_up[feature_cols].values if len(feature_cols) == 178 else (df_up.values if df_up.shape[1] == 178 else None)

                if X_input is None:
                    st.error(f"Expected 178 feature columns, got {df_up.shape[1]}.")
                else:
                    st.success(f"✅ {df_up.shape[0]} samples loaded")
                    if not model_loaded:
                        st.warning("Model not trained yet — showing signal preview only.")
                        for i in range(min(3, len(X_input))):
                            fig = plot_signal(X_input[i], f"Sample {i+1}", '#3498db')
                            st.pyplot(fig); plt.close()
                    else:
                        X_proc = preprocess(X_input)
                        with st.spinner("Running inference..."):
                            probs = _predict(model, X_proc)
                            preds = (probs >= 0.5).astype(int)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total", len(preds))
                        c2.metric("🔴 Seizure", int(preds.sum()))
                        c3.metric("🟢 Normal", int((1-preds).sum()))

                        st.markdown("---")
                        for i in range(min(5, len(X_input))):
                            col_a, col_b = st.columns([4, 1])
                            with col_a:
                                color = '#e74c3c' if preds[i] == 1 else '#2ecc71'
                                fig = plot_signal(X_input[i], f"Sample {i+1}", color, preds[i])
                                st.pyplot(fig); plt.close()
                            with col_b:
                                conf = max(probs[i], 1 - probs[i]) * 100
                                if preds[i] == 1:
                                    st.markdown(f'<div class="result-seizure"><div class="label">⚠️ SEIZURE</div><div class="conf">{conf:.1f}% confidence</div></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="result-normal"><div class="label">✓ NORMAL</div><div class="conf">{conf:.1f}% confidence</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        if df_data is None:
            st.warning("Dataset not found in `data/` folder.")
        else:
            feature_cols = [c for c in df_data.columns if c.startswith('X')]
            col1, col2 = st.columns(2)
            with col1:
                class_sel = st.selectbox("Select class:", list(range(1, 6)),
                    format_func=lambda x: f"Class {x} — {CLASS_NAMES[x]}")
            with col2:
                n_sel = st.slider("Number of samples:", 1, 5, 3)

            if st.button("▶  Run Prediction"):
                subset = df_data[df_data['y'] == class_sel].sample(n_sel, random_state=np.random.randint(1000))
                X_input = subset[feature_cols].values
                true_labels = subset['y'].values

                for i in range(len(X_input)):
                    st.markdown(f"<div style='color:#444;font-size:0.8rem;margin-top:1.5rem;text-transform:uppercase;letter-spacing:0.1em;'>Sample {i+1}</div>", unsafe_allow_html=True)
                    col_a, col_b = st.columns([4, 1])

                    if not model_loaded:
                        color = '#e74c3c' if true_labels[i] == 1 else '#3498db'
                        with col_a:
                            fig = plot_signal(X_input[i], f"True: {CLASS_NAMES[true_labels[i]]}", color)
                            st.pyplot(fig); plt.close()
                        with col_b:
                            st.markdown(f"<div style='color:#555;font-size:0.85rem;'>True label:<br><b style='color:#e0e0e0;'>{CLASS_NAMES[true_labels[i]]}</b></div>", unsafe_allow_html=True)
                    else:
                        X_proc = preprocess(X_input[i:i+1])
                        prob = _predict(model, X_proc)[0]
                        pred = int(prob >= 0.5)
                        conf = max(prob, 1 - prob) * 100
                        color = '#e74c3c' if pred == 1 else '#2ecc71'

                        with col_a:
                            fig = plot_signal(X_input[i], f"True: Class {true_labels[i]} — {CLASS_NAMES[true_labels[i]]}", color, pred)
                            st.pyplot(fig); plt.close()
                        with col_b:
                            if pred == 1:
                                st.markdown(f'<div class="result-seizure"><div class="label">⚠️<br>SEIZURE</div><div class="conf">{conf:.1f}%</div></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="result-normal"><div class="label">✓<br>NORMAL</div><div class="conf">{conf:.1f}%</div></div>', unsafe_allow_html=True)
                            correct = (pred == 1) == (true_labels[i] == 1)
                            st.markdown(f"<div style='margin-top:0.8rem;font-size:0.85rem;color:#555;'>True: <b style='color:#aaa;'>{CLASS_NAMES[true_labels[i]]}</b><br>{'<span style=\"color:#2ecc71;\">✓ Correct</span>' if correct else '<span style=\"color:#e74c3c;\">✗ Wrong</span>'}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA Plots":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    plots = [
        ("EEG Signals Per Class",           "outputs/eda_eeg_signals.png"),
        ("Class Distribution",               "outputs/eda_class_distribution.png"),
        ("Average Signal Shape Per Class",   "outputs/eda_avg_signal_per_class.png"),
        ("PCA Visualization",                "outputs/eda_pca.png"),
        ("Signal Statistics Per Class",      "outputs/eda_signal_stats.png"),
        ("Amplitude Distribution",           "outputs/eda_amplitude_distribution.png"),
        ("Feature Variance Analysis",        "outputs/eda_feature_variance.png"),
        ("Correlation Heatmap (X1–X20)",     "outputs/eda_correlation_heatmap.png"),
    ]

    for i in range(0, len(plots), 2):
        c1, c2 = st.columns(2, gap="medium")
        for col, (title, path) in zip([c1, c2], plots[i:i+2]):
            with col:
                st.markdown(f"<div style='color:#aaa;font-size:0.9rem;font-weight:600;margin-bottom:0.5rem;'>{title}</div>", unsafe_allow_html=True)
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.markdown(f"<div style='background:#111;border:1px dashed #333;border-radius:10px;padding:2rem;text-align:center;color:#444;'>Run eda.ipynb to generate</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  Model Info":
    st.markdown('<div class="section-header">Model Architectures</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    archs = [
        ("🔴 1D CNN", "#e74c3c",
         "Input → (178, 1)\nConv1D(64, k=5) + BN + Pool\nConv1D(128, k=5) + BN + Pool\nConv1D(256, k=3) + BN + Pool\nFlatten\nDense(128) → Dense(64)\nDense(1, sigmoid)",
         "Captures local EEG patterns via 3-layer convolution blocks"),
        ("🔵 LSTM", "#3498db",
         "Input → (178, 1)\nLSTM(128, return_seq=True)\nLSTM(64)\nDense(64)\nDense(1, sigmoid)",
         "Learns long-range temporal dependencies in EEG sequences"),
        ("🟢 CNN-LSTM Hybrid", "#2ecc71",
         "Input → (178, 1)\nConv1D(64) + BN + Pool\nConv1D(128) + BN + Pool\nLSTM(128, return_seq=True)\nLSTM(64)\nDense(64)\nDense(1, sigmoid)",
         "Best of both — local features (CNN) + temporal context (LSTM)"),
    ]
    for col, (name, color, arch, desc) in zip([c1, c2, c3], archs):
        with col:
            st.markdown(f"""
            <div style='background:#111;border:1px solid #222;border-top:3px solid {color};border-radius:14px;padding:1.5rem;'>
                <div style='font-weight:700;color:{color};font-size:1rem;margin-bottom:1rem;'>{name}</div>
                <pre style='background:#0a0a0a;border:1px solid #1a1a1a;border-radius:8px;
                            padding:1rem;font-size:0.78rem;color:#aaa;overflow-x:auto;'>{arch}</pre>
                <div style='color:#555;font-size:0.82rem;margin-top:0.8rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Loss Function | Binary Crossentropy |
        | Optimizer | Adam (lr = 0.001) |
        | Batch Size | 128 |
        | Max Epochs | 30 |
        | Validation Split | 15% |
        """)
    with c2:
        st.markdown("""
        | Callback | Config |
        |---|---|
        | EarlyStopping | patience=8, restore best |
        | ReduceLROnPlateau | factor=0.5, patience=4 |
        | ModelCheckpoint | save best only (.h5) |
        """)

    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    techs = [("🐍", "Python 3.10"), ("🧠", "TensorFlow 2.x"), ("📊", "scikit-learn"),
             ("📈", "Matplotlib"), ("🐼", "Pandas"), ("🌐", "Streamlit")]
    cols = st.columns(len(techs))
    for col, (icon, name) in zip(cols, techs):
        with col:
            st.markdown(f"""
            <div style='background:#111;border:1px solid #222;border-radius:10px;
                        padding:1rem;text-align:center;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='color:#aaa;font-size:0.82rem;margin-top:0.3rem;'>{name}</div>
            </div>""", unsafe_allow_html=True)
