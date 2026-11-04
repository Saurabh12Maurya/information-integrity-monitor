from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sample_data.csv"
MODEL_PATH = ROOT / "review_priority_model.joblib"

st.set_page_config(
    page_title="Information Integrity Monitor",
    page_icon="🛡️",
    layout="wide"
)

NAVY = "#17243A"
PANEL = "#22324A"
OLIVE = "#8A9B70"
TEXT = "#E8EDF4"
MUTED = "#AAB6C5"

st.markdown(f"""
<style>
.stApp {{ background: #101827; color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: #172337; }}
h1, h2, h3, h4, p, label, span {{ color: {TEXT}; }}
small, .stCaption {{ color: {MUTED} !important; }}
div[data-testid="stMetric"] {{
    background: {PANEL}; border: 1px solid #3B4B62;
    padding: 14px; border-radius: 10px;
}}
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
    background: {PANEL}; border-radius: 8px;
}}
.note {{
    background: #26382F; color: #E8F0E4; padding: 13px 16px;
    border-left: 5px solid {OLIVE}; border-radius: 6px;
}}
.stButton button {{ background: {OLIVE}; color: #101827; border-radius: 7px; }}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Information Integrity Monitor")
st.caption("Research prototype | AI/ML in Information Management")
st.markdown(
    '<div class="note"><b>Scope:</b> Demonstration using synthetic examples only. '
    'The model estimates review priority; it does not establish truth or falsehood, '
    'monitor private communications, or issue public statements. All actions require '
    'authorised human review.</div>',
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

st.sidebar.header("Review settings")
velocity_threshold = st.sidebar.slider(
    "Rapid-growth threshold (mentions/hour)", 10, 200, 50, 5
)
impact_weight = st.sidebar.selectbox(
    "Impact weighting", ["Standard", "Conservative"], index=0
)
st.sidebar.caption("Illustrative settings for a classroom prototype; not operational thresholds.")

df["change"] = df["mentions_1h"] - df["mentions_previous_1h"]
df["growth_rate"] = (
    df["change"] / df["mentions_previous_1h"].clip(lower=1)
) * 100

def assess(row):
    score = 0
    if row["mentions_1h"] >= velocity_threshold:
        score += 25
    if row["change"] > 0:
        score += min(25, row["change"] / 2)
    if row["evidence_status"] == "not_checked":
        score += 25
    elif row["evidence_status"] == "contradicted":
        score += 35
    if row["potential_impact"] == "high":
        score += 15
    elif row["potential_impact"] == "medium":
        score += 8
    if impact_weight == "Conservative":
        score *= 0.85
    return int(min(100, round(score)))

def recommendation(row):
    if row["evidence_status"] == "verified":
        return "No correction needed; retain source link"
    if row["evidence_status"] == "contradicted":
        return "Priority human verification; consider approved correction"
    if row["mentions_1h"] >= velocity_threshold and row["potential_impact"] in ("medium", "high"):
        return "Priority verification; do not auto-publish"
    return "Monitor quietly; gather reliable evidence"

df["triage_score"] = df.apply(assess, axis=1)
df["recommendation"] = df.apply(recommendation, axis=1)

if model is not None:
    df["ml_review_priority"] = model.predict(df["claim_text"].astype(str))
    probabilities = model.predict_proba(df["claim_text"].astype(str))
    classes = list(model.classes_)
    if "priority_review" in classes:
        priority_index = classes.index("priority_review")
        df["ml_priority_confidence"] = probabilities[:, priority_index] * 100
    else:
        df["ml_priority_confidence"] = 0.0
else:
    df["ml_review_priority"] = "Model not trained"
    df["ml_priority_confidence"] = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Claims in sample", len(df))
c2.metric("Verified examples", int((df.evidence_status == "verified").sum()))
c3.metric("Needs verification", int((df.evidence_status == "not_checked").sum()))
c4.metric("Priority review", int((df.triage_score >= 60).sum()))

st.subheader("Claim review queue")
st.dataframe(
    df[["claim_id", "claim_text", "mentions_1h", "mentions_previous_1h",
        "source_status", "evidence_status", "potential_impact"]],
    use_container_width=True, hide_index=True
)

st.subheader("Assessment and recommendation")
display = df[["claim_id", "change", "growth_rate", "triage_score",
              "recommendation", "ml_review_priority", "ml_priority_confidence"]].copy()
display["growth_rate"] = display["growth_rate"].round(1).astype(str) + "%"
display["ml_priority_confidence"] = display["ml_priority_confidence"].round(1).astype(str) + "%"
st.dataframe(display, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown("#### Mentions in the latest hour")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#22324A")
    ax.bar(df["claim_id"], df["mentions_1h"], color=OLIVE)
    ax.set_ylabel("Mentions (synthetic)", color=TEXT)
    ax.set_xlabel("Claim", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#66758A")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("#### Rule-based triage score")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#172337")
    ax.set_facecolor("#22324A")
    ax.bar(df["claim_id"], df["triage_score"], color="#7296B8")
    ax.axhline(60, linestyle="--", linewidth=1, color=OLIVE, label="Illustrative priority line")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0–100)", color=TEXT)
    ax.set_xlabel("Claim", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#66758A")
    legend = ax.legend()
    for label in legend.get_texts():
        label.set_color(TEXT)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Decision principles")
st.markdown("""
- **Verify before communicating:** an unverified claim is not automatically false.
- **Avoid amplification:** low-circulation claims may be monitored without public repetition.
- **Escalate proportionately:** higher spread or potential impact triggers human review, not automatic publication.
- **Keep an audit trail:** a real deployment should record evidence, reviewer, decision and timestamp.
""")

if model is None:
    st.warning("The ML model has not been trained yet. Run `python train_model.py` in the project folder, then refresh this page.")
else:
    st.success("Demonstration NLP model loaded. Its output is a review-priority suggestion, not a factuality verdict.")

st.info(
    "The ML model is trained on a small synthetic dataset for demonstration. "
    "Its evaluation score is not evidence of real-world performance. "
    "Use an authorised, representative, human-labelled dataset before any real deployment."
)
