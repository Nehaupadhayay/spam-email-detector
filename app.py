"""
app.py - Email Spam Detector | Final Year Project
Deploy on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import re
from model_utils import (
    build_and_train_model, predict_email, get_spam_features, preprocess_text
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam E-Mail Detector | Email Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global ── */
* { font-family: 'Space Grotesk', sans-serif; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0a0e1a;
}
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c8d6e8 !important; }

/* ── Hide default header ── */
#MainMenu, footer,{ visibility: hidden; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1220; }
::-webkit-scrollbar-thumb { background: #2563eb; border-radius: 3px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a1040 50%, #0f1f3d 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(37,99,235,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 30%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4);
    color: #60a5fa;
    padding: 0.2rem 0.8rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
}

/* ── Cards ── */
.stat-card {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #2563eb; }
.stat-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Result boxes ── */
.result-spam {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(220,38,38,0.04));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.15); }
    50% { box-shadow: 0 0 20px 4px rgba(239,68,68,0.1); }
}
.result-ham {
    background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(16,185,129,0.04));
    border: 2px solid rgba(52,211,153,0.5);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(52,211,153,0.15); }
    50% { box-shadow: 0 0 20px 4px rgba(52,211,153,0.1); }
}
.result-title-spam {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ef4444;
    margin: 0;
}
.result-title-ham {
    font-size: 2.2rem;
    font-weight: 700;
    color: #34d399;
    margin: 0;
}
.result-confidence {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Text area styling ── */
.stTextArea textarea {
    background: #0d1220 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.3) !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #60a5fa;
    letter-spacing: 0.03em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1e3a6e, transparent);
    margin-left: 0.5rem;
}

/* ── Feature pills ── */
.feature-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
}
.pill-high { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.pill-mid  { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.pill-low  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }

/* ── Info box ── */
.info-box {
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1220;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.2rem;
    border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 7px !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a6e, #2d1b69) !important;
    color: #e2e8f0 !important;
}

/* ── Metric labels ── */
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #60a5fa !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Divider ── */
hr { border-color: #1e2d4a !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #0d1220 !important;
    border-color: #1e2d4a !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models, best_model, metrics, X_test, y_test = build_and_train_model()
    return models, best_model, metrics, X_test, y_test


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem;'>
        <div style='font-size:1.5rem; font-weight:700; color:#60a5fa;'>🛡️ Spam E-Mail Detector</div>
        <div style='font-size:0.78rem; color:#475569; margin-top:0.3rem;'>AI Powered Email Security System </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # ── Detection Mode ──
    st.markdown("### ⚙️ Detection Mode")

    mode = st.radio(
        "",
        ["🟢 Safe", "🟡 Balanced", "🔴 Aggressive"],
        horizontal=True
    )

    # Threshold mapping
    if mode == "🟢 Safe":
        threshold = 0.7
        sensitivity = "Low"
    elif mode == "🟡 Balanced":
        threshold = 0.5
        sensitivity = "Medium"
    else:
        threshold = 0.3
        sensitivity = "High"

    # Sensitivity text
    st.markdown(f"""
    <div style='font-size:0.85rem; color:#94a3b8; margin-top:0.3rem;'>
    Spam Sensitivity: <b style='color:#60a5fa;'>{sensitivity}</b>
    </div>
    """, unsafe_allow_html=True)
    # Default model (IMPORTANT)
    selected_model_name = "Logistic Regression"


    # ── Advanced Settings ──
    with st.expander("⚙️ Advanced Settings"):
        selected_model_name = st.selectbox(
            "Model",
            ["Logistic Regression", "Naive Bayes", "Random Forest"]
        )

        st.markdown(f"""
        <div style='font-size:0.8rem; color:#64748b;'>
        Threshold: {threshold}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 Navigation")
        page = st.radio(
            "",
            ["🔍 Spam Detector", "📊 Model Analytics", "📖 About"],
            label_visibility="collapsed"
        )

    st.markdown("""
<div style='background:#0d1220; border:1px solid #1e2d4a; border-radius:10px; padding:0.8rem;'>

<div style='color:#60a5fa; font-weight:600; margin-bottom:0.5rem;'>💡 Quick Tips</div>

<div style='font-size:0.82rem; color:#94a3b8; line-height:1.6;'>

• Paste full email for better accuracy<br>
• Avoid very short text inputs<br>
• Try different detection modes<br>
• Use batch analysis for multiple emails

</div>
</div>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────────────────
with st.spinner("🔧 Initializing AI models..."):
    models, best_model_default, metrics, X_test, y_test = load_models()

active_model = models[selected_model_name]
active_metrics = metrics[selected_model_name]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SPAM DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Spam Detector":

    # Hero
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>v2.0 · AI SECURITY · EMAIL FILTER</div>
        <h1 class='hero-title'>🛡️ Spam E-Mail Detector </h1>
        <p class='hero-subtitle'>Smart email protection powered by AI & machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Top stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{active_metrics['accuracy']*100:.1f}%</div>
            <div class='stat-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{active_metrics['precision']*100:.1f}%</div>
            <div class='stat-label'>Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{active_metrics['f1']*100:.1f}%</div>
            <div class='stat-label'>F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{selected_model_name.split()[0]}</div>
            <div class='stat-label'>Active Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Area ──────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown("<div class='section-header'>✉️ Email Input</div>", unsafe_allow_html=True)

        email_text = st.text_area(
            "",
            placeholder="Paste your email content here...\n\nExample: Congratulations! You've won $1,000,000! Click here to claim your prize immediately!",
            height=220,
            label_visibility="collapsed",
            key="email_input"
        )

        # Sample emails
        st.markdown("<div style='font-size:0.8rem; color:#475569; margin-top:0.5rem; margin-bottom:0.3rem;'>Quick test samples:</div>", unsafe_allow_html=True)
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        with sample_col1:
            if st.button("🚨 Spam Sample", use_container_width=True):
                st.session_state['email_input'] = "CONGRATULATIONS!!! You've been selected as our LUCKY WINNER! Click NOW to claim your FREE $5000 cash prize! Limited time offer! Act immediately! Call 1-800-FREE-CASH!"
                st.rerun()
        with sample_col2:
            if st.button("✅ Ham Sample", use_container_width=True):
                st.session_state['email_input'] = "Hi Sarah, hope you're doing well! Just a reminder about our team meeting tomorrow at 2pm in conference room B. Please bring the Q3 report if you have it ready. Thanks!"
                st.rerun()
        with sample_col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state['email_input'] = ""
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Email", use_container_width=True, type="primary")

    with right_col:
        st.markdown("<div class='section-header'>⚡ Real-time Stats</div>", unsafe_allow_html=True)

        if email_text.strip():
            words = email_text.split()
            chars = len(email_text)
            exclamations = email_text.count('!')
            caps_pct = sum(1 for c in email_text if c.isupper()) / max(chars, 1) * 100
            urls = len(re.findall(r'http\S+|www\S+', email_text.lower()))

            st.metric("📝 Word Count", len(words))
            st.metric("🔤 Characters", chars)
            st.metric("❗ Exclamations", exclamations)
            st.metric("🔠 CAPS Ratio", f"{caps_pct:.1f}%")
            st.metric("🔗 URLs Found", urls)

            # Live mini risk score
            risk = min(100, exclamations * 8 + caps_pct * 0.8 + urls * 15)
            risk_color = "#ef4444" if risk > 60 else "#fbbf24" if risk > 30 else "#34d399"
            st.markdown(f"""
            <div style='margin-top:0.5rem; padding:0.8rem; background:#0d1220; border-radius:10px; border:1px solid #1e2d4a;'>
                <div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;'>Live Risk Estimate</div>
                <div style='font-size:1.5rem; font-weight:700; color:{risk_color}; font-family:"JetBrains Mono",monospace;'>{risk:.0f}/100</div>
                <div style='height:4px; background:#1e2d4a; border-radius:2px; margin-top:0.4rem;'>
                    <div style='height:4px; width:{risk}%; background:{risk_color}; border-radius:2px; transition:width 0.3s;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box' style='text-align:center; padding:2rem 1rem;'>
                <div style='font-size:2.5rem;'>✉️</div>
                <div style='color:#475569; margin-top:0.5rem;'>Paste an email to see live statistics</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Analysis Result ─────────────────────────────────────────────────────
    if analyze_btn:
        # ❗ Empty input check
        if not email_text.strip():
            st.warning("⚠️ Please enter email content")
            st.stop()

        # ❗ Short text check
        if len(email_text.split()) < 3:
            st.warning("⚠️ Please enter a meaningful email (at least 3 words)")
            st.stop()

        with st.spinner("🧠 Analyzing email..."):
            time.sleep(0.4)

        # ✅ Clean text
        clean_text = preprocess_text(email_text)

        # ML prediction clean text pe
        result = predict_email(clean_text, active_model)

        # 🔥 IMPORTANT: features raw text pe nikalo
        features = get_spam_features(email_text)
        result['features'] = features
        
        st.write(result['features'])
        st.write(result['spam_probability'])

        # ✅ Strong logic
    # ✅ Final smart scoring logic
        spam_score = 0

        if result['spam_probability'] >= threshold:
            spam_score += 2
        if result['features']['spam_keyword_count'] >= 2:
            spam_score += 1
        if result['features']['url_count'] >= 1:
            spam_score += 1
        if result['features']['exclamation_count'] >= 3:
            spam_score += 1
        if result['features']['caps_ratio'] > 0.2:
            spam_score += 1

        is_spam = spam_score >= 2

        result['prediction'] = 1 if is_spam else 0
        result['label'] = 'SPAM' if is_spam else 'NOT SPAM'

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Detection Result</div>", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 2], gap="large")

        with res_col1:
            if is_spam:
                st.markdown(f"""
                <div class='result-spam'>
                    <div style='font-size:3rem;'>🚨</div>
                    <div class='result-title-spam'>SPAM</div>
                    <div class='result-confidence'>Confidence: {result['confidence']*100:.1f}%</div>
                    <div style='font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;'>Model: {selected_model_name}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-ham'>
                    <div style='font-size:3rem;'>✅</div>
                    <div class='result-title-ham'>NOT SPAM</div>
                    <div class='result-confidence'>Confidence: {result['confidence']*100:.1f}%</div>
                    <div style='font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;'>Model: {selected_model_name}</div>
                </div>
                """, unsafe_allow_html=True)
                

        with res_col2:
            # Probability gauge
            spam_p = result['spam_probability'] * 100
            ham_p = result['ham_probability'] * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=spam_p,
                number={'suffix': "%", 'font': {'size': 28, 'color': '#e2e8f0', 'family': 'JetBrains Mono'}},
                title={'text': "Spam Probability", 'font': {'size': 14, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color': '#475569', 'size': 10}},
                    'bar': {'color': '#ef4444' if spam_p > 50 else '#34d399'},
                    'bgcolor': '#0d1220',
                    'bordercolor': '#1e2d4a',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(52,211,153,0.1)'},
                        {'range': [30, 60], 'color': 'rgba(251,191,36,0.1)'},
                        {'range': [60, 100], 'color': 'rgba(239,68,68,0.1)'},
                    ],
                    'threshold': {
                        'line': {'color': '#fbbf24', 'width': 2},
                        'thickness': 0.75,
                        'value': threshold * 100,
                    }
                }
            ))
            fig.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Space Grotesk'),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature analysis
        st.markdown("<br>", unsafe_allow_html=True)
        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.markdown("<div class='section-header'>🔬 Spam Indicators</div>", unsafe_allow_html=True)
            features = result['features']

            indicators = []
            if features['spam_keyword_count'] > 3:
                indicators.append(('pill-high', f"⚠️ {features['spam_keyword_count']} spam keywords"))
            elif features['spam_keyword_count'] > 0:
                indicators.append(('pill-mid', f"🔸 {features['spam_keyword_count']} spam keywords"))
            else:
                indicators.append(('pill-low', '✅ No spam keywords'))

            if features['exclamation_count'] > 2:
                indicators.append(('pill-high', f"⚠️ {features['exclamation_count']} exclamations"))
            elif features['exclamation_count'] > 0:
                indicators.append(('pill-mid', f"🔸 {features['exclamation_count']} exclamations"))

            if features['caps_ratio'] > 0.3:
                indicators.append(('pill-high', f"⚠️ {features['caps_ratio']*100:.0f}% CAPS"))
            elif features['caps_ratio'] > 0.1:
                indicators.append(('pill-mid', f"🔸 {features['caps_ratio']*100:.0f}% CAPS"))

            if features['url_count'] > 0:
                indicators.append(('pill-mid', f"🔗 {features['url_count']} URL(s)"))

            if features['money_mentions'] > 0:
                indicators.append(('pill-high', f"💰 {features['money_mentions']} money mention(s)"))

            pill_html = "".join([f"<span class='feature-pill {cls}'>{label}</span>" for cls, label in indicators])
            st.markdown(f"<div style='line-height:2.2;'>{pill_html}</div>", unsafe_allow_html=True)

        with feat_col2:
            st.markdown("<div class='section-header'>📊 Probability Breakdown</div>", unsafe_allow_html=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=['NOT SPAM', 'SPAM'],
                y=[ham_p, spam_p],
                marker_color=['rgba(52,211,153,0.8)', 'rgba(239,68,68,0.8)'],
                marker_line_color=['#34d399', '#ef4444'],
                marker_line_width=1,
                text=[f'{ham_p:.1f}%', f'{spam_p:.1f}%'],
                textposition='auto',
                textfont={'color': 'white', 'family': 'JetBrains Mono'},
            ))
            fig2.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=10, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2d4a'),
                yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2d4a', range=[0, 110]),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif analyze_btn and not email_text.strip():
        st.warning("⚠️ Please enter email content before analyzing.")

    # ── Batch Analysis ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Batch Email Analysis — Test Multiple Emails at Once"):
        st.markdown("""
        <div class='info-box'>Enter one email per line below to analyze multiple emails simultaneously.</div>
        """, unsafe_allow_html=True)

        batch_input = st.text_area(
            "Multiple emails (one per line):",
            height=150,
            placeholder="Email 1 text here\nEmail 2 text here\nEmail 3 text here",
        )
        if st.button("🔍 Run Batch Analysis", use_container_width=True):
            if batch_input.strip():
                emails = [e.strip() for e in batch_input.strip().split('\n') if e.strip()]
                results_data = []
                prog = st.progress(0)
                for i, em in enumerate(emails):
                    r = predict_email(em, active_model)
                    is_sp = r['spam_probability'] >= threshold
                    results_data.append({
                        'Email (preview)': em[:60] + '...' if len(em) > 60 else em,
                        'Result': '🚨 SPAM' if is_sp else '✅ HAM',
                        'Spam Prob.': f"{r['spam_probability']*100:.1f}%",
                        'Confidence': f"{r['confidence']*100:.1f}%",
                    })
                    prog.progress((i + 1) / len(emails))

                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, use_container_width=True, hide_index=True)

                spam_count = sum(1 for r in results_data if '🚨' in r['Result'])
                st.markdown(f"""
                <div style='text-align:center; padding:0.8rem; background:#0d1220; border-radius:10px; border:1px solid #1e2d4a; margin-top:0.5rem;'>
                    <b style='color:#ef4444;'>🚨 {spam_count} SPAM</b> &nbsp;|&nbsp; 
                    <b style='color:#34d399;'>✅ {len(emails)-spam_count} HAM</b> &nbsp;|&nbsp; 
                    <b style='color:#60a5fa;'>📧 {len(emails)} Total</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter at least one email.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>ANALYTICS DASHBOARD</div>
        <h1 class='hero-title'>📊 Model Analytics</h1>
        <p class='hero-subtitle'>Deep dive into model performance, comparisons, and evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison Table
    st.markdown("<div class='section-header'>🏆 Model Performance Comparison</div>", unsafe_allow_html=True)

    model_data = []
    for name, m in metrics.items():
        model_data.append({
            'Model': name,
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Precision': f"{m['precision']*100:.2f}%",
            'Recall': f"{m['recall']*100:.2f}%",
            'F1 Score': f"{m['f1']*100:.2f}%",
        })
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    # Radar / bar charts
    tab1, tab2, tab3 = st.tabs(["📈 Performance Bar Chart", "🕸️ Radar Chart", "🔢 Confusion Matrix"])

    with tab1:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig = go.Figure()
        colors = ['#60a5fa', '#34d399', '#f59e0b']
        for i, (name, m) in enumerate(metrics.items()):
            vals = [m['accuracy']*100, m['precision']*100, m['recall']*100, m['f1']*100]
            fig.add_trace(go.Bar(
                name=name,
                x=metric_names,
                y=vals,
                marker_color=colors[i],
                marker_opacity=0.85,
                text=[f'{v:.1f}%' for v in vals],
                textposition='outside',
                textfont={'size': 11, 'color': 'white'},
            ))
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2d4a'),
            yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2d4a', range=[0, 115]),
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig = go.Figure()
        colors = ['#60a5fa', '#34d399', '#f59e0b']
        for i, (name, m) in enumerate(metrics.items()):
            vals = [m['accuracy']*100, m['precision']*100, m['recall']*100, m['f1']*100]
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=name,
                line_color=colors[i],
                fillcolor=colors[i].replace('#', 'rgba(').replace(')', ',0.1)') if False else f'rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)',
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 105], tickfont=dict(color='#475569'), gridcolor='#1e2d4a'),
                angularaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='#1e2d4a'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=420,
            legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cm_col1, cm_col2, cm_col3 = st.columns(3)
        cm_cols = [cm_col1, cm_col2, cm_col3]

        for i, (name, m) in enumerate(metrics.items()):
            cm = np.array(m['confusion_matrix'])
            with cm_cols[i]:
                st.markdown(f"<div style='text-align:center; color:#60a5fa; font-weight:600; margin-bottom:0.5rem;'>{name}</div>", unsafe_allow_html=True)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale=['#0d1220', '#1e3a6e', '#2563eb'],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['HAM', 'SPAM'],
                    y=['HAM', 'SPAM'],
                )
                fig_cm.update_layout(
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(color='#94a3b8')),
                    yaxis=dict(tickfont=dict(color='#94a3b8')),
                    font=dict(color='white'),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    # Dataset Info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📦 Training Dataset Overview</div>", unsafe_allow_html=True)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-value'>80</div>
            <div class='stat-label'>Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col2:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-value'>40</div>
            <div class='stat-label'>Spam Emails</div>
        </div>
        """, unsafe_allow_html=True)
    with d_col3:
        st.markdown("""
        <div class='stat-card'>
            <div class='stat-value'>40</div>
            <div class='stat-label'>Ham Emails</div>
        </div>
        """, unsafe_allow_html=True)

    # Class distribution pie
    fig_pie = go.Figure(go.Pie(
        labels=['Spam', 'Ham'],
        values=[40, 40],
        hole=0.5,
        marker=dict(colors=['rgba(239,68,68,0.8)', 'rgba(52,211,153,0.8)'],
                    line=dict(color=['#ef4444', '#34d399'], width=2)),
        textfont=dict(size=13, color='white'),
    ))
    fig_pie.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=20, b=10),
        annotations=[dict(text='50/50<br>Balance', x=0.5, y=0.5,
                          font=dict(size=13, color='#94a3b8'), showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-badge'>EMail Security System</div>
        <h1 class='hero-title'>📖 About Spam E-Mail Detector</h1>
        <p class='hero-subtitle'>Email Spam Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:#0d1220; border:1px solid #1e2d4a; border-radius:14px; padding:1.8rem;'>
            <div style='color:#60a5fa; font-size:1.1rem; font-weight:600; margin-bottom:1rem;'>🎯 Project Overview</div>
            <p style='color:#94a3b8; line-height:1.8; font-size:0.9rem;'>
                Spam E-Mail Detector is a machine learning-powered email spam detection system built for safety purpose project. It combines NLP preprocessing with multiple ML classifiers to accurately 
                identify spam emails in real-time.
            </p>
            <br/>
            <div style='color:#60a5fa; font-size:1.1rem; font-weight:600; margin-bottom:1rem;'>🛠️ Tech Stack</div>
            <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
                <span class='feature-pill pill-low'>Python 3.10</span>
                <span class='feature-pill pill-low'>Streamlit</span>
                <span class='feature-pill pill-low'>scikit-learn</span>
                <span class='feature-pill pill-low'>NLTK</span>
                <span class='feature-pill pill-low'>Plotly</span>
                <span class='feature-pill pill-low'>Pandas / NumPy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#0d1220; border:1px solid #1e2d4a; border-radius:14px; padding:1.8rem;'>
            <div style='color:#60a5fa; font-size:1.1rem; font-weight:600; margin-bottom:1rem;'>🤖 ML Models Used</div>
            <div style='color:#94a3b8; font-size:0.9rem; line-height:2;'>
                <b style='color:#e2e8f0;'>1. Naive Bayes</b> — Probabilistic classifier based on Bayes' theorem, ideal for text classification.<br/>
                <b style='color:#e2e8f0;'>2. Logistic Regression</b> — Linear classifier excellent for binary classification problems.<br/>
                <b style='color:#e2e8f0;'>3. Random Forest</b> — Ensemble method using multiple decision trees for robust classification.<br/>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1220; border:1px solid #1e2d4a; border-radius:14px; padding:1.8rem;'>
        <div style='color:#60a5fa; font-size:1.1rem; font-weight:600; margin-bottom:1rem;'>⚙️ NLP Pipeline</div>
        <div style='display:flex; align-items:center; gap:0.8rem; flex-wrap:wrap; color:#94a3b8; font-size:0.85rem;'>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem; color:#60a5fa;'>📥 Raw Email Text</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>Lowercase</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>URL/Email Tagging</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>Punctuation Removal</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>Stopword Removal</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>Stemming (Porter)</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem;'>TF-IDF (bigrams)</div>
            <div>→</div>
            <div style='background:#1e2d4a; border-radius:8px; padding:0.5rem 1rem; color:#34d399;'>🎯 Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d1220; border:1px solid #1e2d4a; border-radius:14px; padding:1.8rem;'>
        <div style='color:#60a5fa; font-size:1.1rem; font-weight:600; margin-bottom:1rem;'>✨ Key Features</div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; font-size:0.88rem; color:#94a3b8;'>
            <div>🔍 Real-time spam detection with probability scores</div>
            <div>📊 Detailed spam indicator feature analysis</div>
            <div>🤖 3 ML models with switchable inference</div>
            <div>📋 Batch analysis for multiple emails at once</div>
            <div>📈 Interactive model performance analytics</div>
            <div>🎚️ Adjustable detection threshold control</div>
            <div>⚡ Live character/word/CAPS statistics</div>
            <div>🌐 Fully deployable on Streamlit Cloud</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='text-align:center; color:#475569; font-size:0.78rem; font-family:"JetBrains Mono",monospace;'>
        Built with ❤️ using Python & Streamlit · 
    </div>
    """, unsafe_allow_html=True)
