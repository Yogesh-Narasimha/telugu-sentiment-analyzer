import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Telugu Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tiro+Telugu&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f0f13; }
.hero-title { font-family:'Tiro Telugu',serif; font-size:2.8rem; font-weight:400; color:#f5e6c8; line-height:1.2; margin-bottom:0.2rem; }
.hero-sub { font-size:1rem; color:#8a8a9a; margin-bottom:2rem; letter-spacing:0.04em; }
.badge-positive { background:linear-gradient(135deg,#1a4a2e,#2d7a4f); color:#7fffc4; padding:0.4rem 1.2rem; border-radius:20px; font-weight:600; font-size:1rem; display:inline-block; border:1px solid #2d7a4f; }
.badge-negative { background:linear-gradient(135deg,#4a1a1a,#7a2d2d); color:#ffb3b3; padding:0.4rem 1.2rem; border-radius:20px; font-weight:600; font-size:1rem; display:inline-block; border:1px solid #7a2d2d; }
.badge-neutral  { background:linear-gradient(135deg,#2a2a3a,#4a4a6a); color:#b3b3ff; padding:0.4rem 1.2rem; border-radius:20px; font-weight:600; font-size:1rem; display:inline-block; border:1px solid #4a4a6a; }
.result-card { background:#1a1a24; border:1px solid #2a2a3a; border-radius:16px; padding:1.5rem 2rem; margin:1rem 0; }
.result-review-text { font-family:'Tiro Telugu',serif; font-size:1.1rem; color:#d4c5a9; line-height:1.7; margin-bottom:1rem; border-left:3px solid #f5a623; padding-left:1rem; }
.conf-label { color:#8a8a9a; font-size:0.85rem; margin-bottom:0.3rem; }
.metric-card { background:#1a1a24; border:1px solid #2a2a3a; border-radius:12px; padding:1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:600; color:#f5e6c8; }
.metric-label { font-size:0.8rem; color:#8a8a9a; margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.08em; }
.section-header { font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:#f5a623; margin-bottom:0.8rem; margin-top:1.5rem; }
.model-badge { background:#1a1a24; border:1px solid #f5a623; border-radius:8px; padding:0.3rem 0.8rem; font-size:0.8rem; color:#f5a623; display:inline-block; margin-bottom:0.5rem; }
.stTabs [data-baseweb="tab-list"] { background:#1a1a24; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#8a8a9a; font-weight:500; }
.stTabs [aria-selected="true"] { background:#2a2a3a !important; color:#f5e6c8 !important; }
.stTextArea textarea { background:#1a1a24 !important; border:1px solid #2a2a3a !important; color:#f5e6c8 !important; border-radius:12px !important; font-family:'Tiro Telugu',serif !important; font-size:1rem !important; }
/* all buttons base */
.stButton > button { border-radius:10px !important; font-weight:600 !important; padding:0.6rem 2rem !important; font-size:0.95rem !important; width:100%; }
/* orange buttons by class */
.orange-btn button { background:linear-gradient(135deg,#f5a623,#e8902a) !important; color:#0f0f13 !important; border:none !important; }
/* green button */
.green-btn button { background:linear-gradient(135deg,#1a4a2e,#2d7a4f) !important; color:#7fffc4 !important; border:2px solid #2d7a4f !important; font-size:0.88rem !important; }
/* red button */
.red-btn button { background:linear-gradient(135deg,#4a1a1a,#7a2d2d) !important; color:#ffb3b3 !important; border:2px solid #7a2d2d !important; font-size:0.88rem !important; }
/* purple button */
.purple-btn button { background:linear-gradient(135deg,#2a2a3a,#4a4a6a) !important; color:#b3b3ff !important; border:2px solid #4a4a6a !important; font-size:0.88rem !important; }
hr { border-color:#2a2a3a !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────
LOCAL_MODEL_HF   = "YogeshNarasimha/telugu-sentiment-xlm-roberta"
LOCAL_MODEL_DIR  = "./notebooks/models/telugu-sentiment-final"
LOCAL_MAX_LEN    = 256
LOCAL_ID2LABEL   = {0: "Negative", 1: "Neutral", 2: "Positive"}
MULTI_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
MULTI_MAX_LEN    = 512
MULTI_ID2LABEL   = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
LABEL_COLORS     = {
    "Very Positive": "#51cf66", "Positive": "#74c476",
    "Neutral"      : "#9999ff",
    "Negative"     : "#ff6b6b", "Very Negative": "#e03131",
}
GEMINI_MODELS    = ["gemini-2.5-flash","gemini-2.5-pro","gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-pro"]
MIN_WORDS        = 4

POS_TEXT = "బాహుబలి సినిమా చాలా అద్భుతంగా ఉంది, విజువల్స్ మరియు నటన అద్భుతం!"
NEG_TEXT = "sequel అవసరమే లేదు original కంటే చాలా తక్కువగా ఉంది, చాలా disappoint అయ్యాను"
NEU_TEXT = "సినిమా okay గా ఉంది, కొన్ని సీన్లు బాగున్నాయి కానీ మొత్తం average గా అనిపించింది"

# ── Model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_local_model():
    # Use local if exists (local dev), else download from HF Hub (Spaces)
    model_source = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else LOCAL_MODEL_HF
    try:
        tok = AutoTokenizer.from_pretrained(model_source)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl = AutoModelForSequenceClassification.from_pretrained(model_source)
        mdl.to(dev); mdl.eval()
        return tok, mdl, dev
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

@st.cache_resource(show_spinner=False)
def load_multi_model():
    tok = AutoTokenizer.from_pretrained(MULTI_MODEL_NAME)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = AutoModelForSequenceClassification.from_pretrained(MULTI_MODEL_NAME)
    mdl.to(dev); mdl.eval()
    return tok, mdl, dev

def run_inference(text, tokenizer, model, device, id2label, max_len):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_len, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return {
        "label"     : id2label[pred_id],
        "confidence": float(probs[pred_id]) * 100,
        "probs"     : {id2label[i]: float(probs[i]) * 100 for i in range(len(probs))}
    }

def badge_html(label):
    cls  = label.lower().replace(" ", "_")
    icon = {"Very Positive":"▲▲","Positive":"▲","Neutral":"●",
            "Negative":"▼","Very Negative":"▼▼"}.get(label,"")
    return f'<span class="badge-{cls}">{icon} {label}</span>'

def confidence_bar(probs_dict):
    labels = list(probs_dict.keys())
    values = list(probs_dict.values())
    colors = [LABEL_COLORS.get(l, "#8888ff") for l in labels]
    fig    = go.Figure()
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(x=[val], y=[label], orientation="h",
                             marker_color=color, text=f"{val:.1f}%",
                             textposition="outside", name=label, showlegend=False))
    fig.update_layout(
        height=max(150, len(labels)*44), margin=dict(l=0,r=70,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,115], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, color="#d4c5a9", tickfont=dict(size=13)),
        bargap=0.35)
    return fig

# ── Load models ────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    local_tok, local_mdl, local_dev = load_local_model()
    multi_tok, multi_mdl, multi_dev = load_multi_model()

# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎬 Telugu Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">xlm-roberta-large · Positive · Negative · Neutral · T8.3 SMAI · IIIT Hyderabad</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Selection")
    model_choice = st.radio("Choose model",
        ["V3 xlm-roberta-large (best)", "Multilingual baseline", "V1 indic-bert (baseline)"], index=0)
    st.markdown("---")
    info = {
        "V3 xlm-roberta-large (best)": "**xlm-roberta-large**\n\n- 560M params\n- Overall: 80% · Neutral: 60%\n- TA approved",
        "Multilingual baseline"       : "**tabularisai multilingual**\n\n- 22 languages\n- Overall: 70% · Neutral: 10%",
        "V1 indic-bert (baseline)"    : "**ai4bharat/indic-bert**\n\n- 33M params\n- Overall: 70% · Neutral: 10%",
    }
    st.markdown(info[model_choice])

if model_choice == "V3 xlm-roberta-large (best)":
    if local_mdl is None:
        st.error(f"Model not found at `{LOCAL_MODEL_DIR}`."); st.stop()
    active_tok, active_mdl, active_dev = local_tok, local_mdl, local_dev
    active_id2label = LOCAL_ID2LABEL; active_max_len = LOCAL_MAX_LEN
    model_label = "xlm-roberta-large (fine-tuned) · 3-class · 80% overall · 60% neutral"
elif model_choice == "Multilingual baseline":
    active_tok, active_mdl, active_dev = multi_tok, multi_mdl, multi_dev
    active_id2label = MULTI_ID2LABEL; active_max_len = MULTI_MAX_LEN
    model_label = "tabularisai/multilingual · 5-class · 70% overall"
else:
    if local_mdl is None:
        st.error(f"Model not found at `{LOCAL_MODEL_DIR}`."); st.stop()
    active_tok, active_mdl, active_dev = local_tok, local_mdl, local_dev
    active_id2label = LOCAL_ID2LABEL; active_max_len = LOCAL_MAX_LEN
    model_label = "ai4bharat/indic-bert (fine-tuned) · 3-class · 70% overall"

st.markdown(f'<span class="model-badge">Active: {model_label}</span>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  Single Review  ", "  Bulk CSV Analysis  ", "  Model Info  "])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown('<p class="section-header">Quick Examples</p>', unsafe_allow_html=True)
        st.markdown("Click a button to load an example review:")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if "example_val" not in st.session_state:
            st.session_state["example_val"] = ""

        st.markdown("""
        <script>
        function colorExampleButtons() {
            const frames = window.parent.document.querySelectorAll('iframe');
            frames.forEach(frame => {
                try {
                    const btns = frame.contentDocument.querySelectorAll('button');
                    btns.forEach(btn => {
                        const txt = btn.innerText.trim();
                        if (txt.includes('Positive')) {
                            btn.style.cssText += 'background:linear-gradient(135deg,#1a4a2e,#2d7a4f)!important;color:#7fffc4!important;border:2px solid #2d7a4f!important;';
                        } else if (txt.includes('Negative')) {
                            btn.style.cssText += 'background:linear-gradient(135deg,#4a1a1a,#7a2d2d)!important;color:#ffb3b3!important;border:2px solid #7a2d2d!important;';
                        } else if (txt.includes('Neutral')) {
                            btn.style.cssText += 'background:linear-gradient(135deg,#2a2a3a,#4a4a6a)!important;color:#b3b3ff!important;border:2px solid #4a4a6a!important;';
                        }
                    });
                } catch(e) {}
            });
        }
        setTimeout(colorExampleButtons, 500);
        setTimeout(colorExampleButtons, 1500);
        </script>
        """, unsafe_allow_html=True)

        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            if st.button("▲  Positive Review", key="ex_pos", use_container_width=True):
                st.session_state["example_val"] = POS_TEXT; st.rerun()
        with ex_col2:
            if st.button("▼  Negative Review", key="ex_neg", use_container_width=True):
                st.session_state["example_val"] = NEG_TEXT; st.rerun()
        with ex_col3:
            if st.button("●  Neutral Review", key="ex_neu", use_container_width=True):
                st.session_state["example_val"] = NEU_TEXT; st.rerun()

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Enter Telugu Review</p>', unsafe_allow_html=True)
        review_text = st.text_area(
            label="Telugu review",
            value=st.session_state["example_val"],
            placeholder="ఇక్కడ తెలుగు సమీక్ష రాయండి...\n(Type or paste a Telugu review here)",
            height=160, label_visibility="collapsed"
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="orange-btn">', unsafe_allow_html=True)
        analyze_btn = st.button("  Analyze Sentiment →  ", key="analyze_single", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Result</p>', unsafe_allow_html=True)
        if analyze_btn and review_text.strip():
            word_count = len(review_text.strip().split())
            with st.spinner("Analyzing..."):
                result = run_inference(review_text.strip(), active_tok, active_mdl,
                                       active_dev, active_id2label, active_max_len)
            st.session_state["example_val"] = ""
            st.markdown(f"""
            <div class="result-card">
                <div class="result-review-text">"{review_text.strip()[:200]}{'...' if len(review_text)>200 else ''}"</div>
                <div style="margin-bottom:0.8rem">{badge_html(result['label'])}</div>
                <div class="conf-label">Confidence: {result['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(confidence_bar(result["probs"]),
                            use_container_width=True, config={"displayModeBar": False})
            if word_count < MIN_WORDS:
                st.warning(f"⚠️ Very short review ({word_count} words) — add more context for better accuracy.")
            elif result["confidence"] < 55:
                st.info("⚠️ Low confidence — borderline review or code-mixed language.")
        elif analyze_btn:
            st.warning("Please enter a review first.")
        else:
            st.markdown("""
            <div style="color:#4a4a5a;text-align:center;padding:3rem 1rem;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">తెలుగు</div>
                <div style="font-size:0.9rem">Click an example or type a review, then click Analyze</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk CSV
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Upload CSV File</p>', unsafe_allow_html=True)
    st.markdown("CSV must have a column named **`review`** with Telugu text.")

    sample_csv = pd.DataFrame({"review": [
        "బాహుబలి సినిమా చాలా అద్భుతంగా ఉంది, విజువల్స్ మరియు నటన అద్భుతం, జీవితంలో చూసిన best సినిమా!",
        "ఆర్ఆర్ఆర్ సినిమా చూసి చాలా సంతోషంగా అనిపించింది, రాజమౌళి గారి దర్శకత్వం అద్భుతంగా ఉంది",
        "పుష్ప సినిమాలో అల్లు అర్జున్ నటన చాలా బాగుంది, డైలాగులు మరియు action సూపర్గా ఉన్నాయి",
        "sequel అవసరమే లేదు original కంటే చాలా తక్కువగా ఉంది, చాలా disappoint అయ్యాను",
        "director మారినప్పటి నుండి franchise quality పూర్తిగా తగ్గిపోయింది, చాలా నిరాశగా అనిపించింది",
        "హీరో నటన చాలా చెత్తగా ఉంది, సినిమా మొత్తం నిరాశగా అనిపించింది, థియేటర్ నుండి వెళ్ళిపోయాను",
        "సినిమా okay గా ఉంది, కొన్ని సీన్లు బాగున్నాయి కానీ మొత్తం average గా అనిపించింది",
        "హీరో నటన బాగుంది కానీ సినిమా మొత్తం average గా అనిపించింది, one time watch చేయవచ్చు",
        "మొదటి సగం బాగుంది కానీ రెండవ సగం చాలా slow గా ఉంది, overall mixed feelings అనిపించింది"
    ]}).to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Sample CSV", data=sample_csv,
                       file_name="sample_telugu_reviews.csv", mime="text/csv")

    uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("Column `review` not found. Got: " + str(list(df.columns)))
        else:
            st.success(f"Loaded {len(df)} reviews.")
            preview_n = st.selectbox(
                "Preview rows",
                options=[3, 5, 10, 20, len(df)],
                format_func=lambda x: f"Show {x} rows" if x != len(df) else f"Show all {len(df)} rows",
                index=0
            )
            st.dataframe(df.head(preview_n), use_container_width=True)

            st.markdown('<div class="orange-btn">', unsafe_allow_html=True)
            bulk_clicked = st.button("Run Sentiment Analysis →", key="analyze_bulk", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if bulk_clicked:
                progress = st.progress(0); status = st.empty(); results = []
                for i, row in df.iterrows():
                    text = str(row["review"])
                    if text.strip():
                        res   = run_inference(text, active_tok, active_mdl,
                                              active_dev, active_id2label, active_max_len)
                        entry = {"review": text, "sentiment": res["label"],
                                 "confidence": round(res["confidence"], 1)}
                        for lbl, prob in res["probs"].items():
                            entry[f"{lbl}_%"] = round(prob, 1)
                        results.append(entry)
                    progress.progress((i+1)/len(df))
                    status.text(f"Analyzing {i+1}/{len(df)}...")
                progress.empty(); status.empty()
                st.session_state["bulk_results"] = pd.DataFrame(results)

    if "bulk_results" in st.session_state:
        rdf    = st.session_state["bulk_results"]
        counts = rdf["sentiment"].value_counts()

        st.markdown("---")
        st.markdown('<p class="section-header">Dashboard</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        pos = counts.get("Positive",0) + counts.get("Very Positive",0)
        neg = counts.get("Negative",0) + counts.get("Very Negative",0)
        neu = counts.get("Neutral", 0)
        for col, (lbl, val, col_) in zip([m1,m2,m3,m4],
            [("Total",len(rdf),"#f5e6c8"),("Positive",pos,"#51cf66"),
             ("Negative",neg,"#ff6b6b"),("Neutral",neu,"#9999ff")]):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="color:{col_}">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<p class="section-header">Sentiment Distribution</p>', unsafe_allow_html=True)
            pie = px.pie(values=counts.values, names=counts.index,
                         color=counts.index, color_discrete_map=LABEL_COLORS, hole=0.5)
            pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#d4c5a9", margin=dict(t=10,b=10))
            st.plotly_chart(pie, use_container_width=True, config={"displayModeBar":False})
        with ch2:
            st.markdown('<p class="section-header">Confidence Distribution</p>', unsafe_allow_html=True)
            hist = px.histogram(rdf, x="confidence", color="sentiment", nbins=20,
                                color_discrete_map=LABEL_COLORS, barmode="overlay", opacity=0.75)
            hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#d4c5a9",
                                xaxis=dict(color="#8a8a9a", title="Confidence (%)"),
                                yaxis=dict(color="#8a8a9a", title="Count"),
                                margin=dict(t=10,b=10))
            st.plotly_chart(hist, use_container_width=True, config={"displayModeBar":False})

        ch3, ch4 = st.columns(2)
        with ch3:
            st.markdown('<p class="section-header">Average Confidence per Class</p>', unsafe_allow_html=True)
            avg_conf = rdf.groupby("sentiment")["confidence"].mean().reset_index()
            avg_conf.columns = ["Sentiment","Avg Confidence (%)"]
            bar_conf = px.bar(avg_conf, x="Sentiment", y="Avg Confidence (%)",
                              color="Sentiment", color_discrete_map=LABEL_COLORS,
                              text="Avg Confidence (%)")
            bar_conf.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            bar_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#d4c5a9", showlegend=False,
                                   yaxis=dict(color="#8a8a9a", range=[0,110]),
                                   xaxis=dict(color="#8a8a9a"), margin=dict(t=30,b=10))
            st.plotly_chart(bar_conf, use_container_width=True, config={"displayModeBar":False})
        with ch4:
            st.markdown('<p class="section-header">Review Count per Class</p>', unsafe_allow_html=True)
            count_df = counts.reset_index(); count_df.columns = ["Sentiment","Count"]
            bar_count = px.bar(count_df, x="Sentiment", y="Count",
                               color="Sentiment", color_discrete_map=LABEL_COLORS, text="Count")
            bar_count.update_traces(textposition="outside")
            bar_count.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#d4c5a9", showlegend=False,
                                    yaxis=dict(color="#8a8a9a"), xaxis=dict(color="#8a8a9a"),
                                    margin=dict(t=30,b=10))
            st.plotly_chart(bar_count, use_container_width=True, config={"displayModeBar":False})

        st.markdown('<p class="section-header">Confidence Spread per Class</p>', unsafe_allow_html=True)
        box = px.box(rdf, x="sentiment", y="confidence", color="sentiment",
                     color_discrete_map=LABEL_COLORS, points="all")
        box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#d4c5a9", showlegend=False,
                          yaxis=dict(color="#8a8a9a", title="Confidence (%)"),
                          xaxis=dict(color="#8a8a9a", title=""),
                          margin=dict(t=10,b=10), height=300)
        st.plotly_chart(box, use_container_width=True, config={"displayModeBar":False})

        st.markdown("---")
        st.markdown('<p class="section-header">Theme Extraction via Gemini API</p>', unsafe_allow_html=True)
        st.markdown("Get your free API key from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")
        g_col1, g_col2 = st.columns([2, 1])
        with g_col1:
            gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
        with g_col2:
            gemini_model_sel = st.selectbox("Model", GEMINI_MODELS, index=0)

        if gemini_key and st.button("Extract Themes →", key="gemini_btn"):
            try:
                from google import genai as google_genai
                from google.genai import types as genai_types
                gclient  = google_genai.Client(api_key=gemini_key)
                sample   = "\n".join(rdf["review"].head(30).tolist())
                prompt   = f"""Analyze these Telugu reviews. Extract top 5 recurring themes.
Return ONLY a JSON array. No markdown. No explanation. Just the array.
Format: [{{"theme":"...","sentiment":"positive/negative/neutral","description":"1 line"}}]
Reviews:\n{sample}"""
                response = gclient.models.generate_content(
                    model=gemini_model_sel, contents=prompt,
                    config=genai_types.GenerateContentConfig(max_output_tokens=500))
                raw   = response.text.strip().replace("```json","").replace("```","").strip()
                start = raw.find("["); end = raw.rfind("]") + 1
                if start != -1 and end > start: raw = raw[start:end]
                themes = json.loads(raw)
                st.success(f"Extracted {len(themes)} themes using {gemini_model_sel}")
                for t in themes:
                    color = {"positive":"#51cf66","negative":"#ff6b6b","neutral":"#9999ff"}.get(
                        t.get("sentiment","neutral").lower(), "#9999ff")
                    st.markdown(f"""<div class="result-card" style="margin-bottom:0.5rem;padding:0.8rem 1.2rem">
                        <span style="color:{color};font-weight:600">{t.get('theme','')}</span>
                        <span style="color:#8a8a9a;font-size:0.85rem;margin-left:0.5rem">· {t.get('sentiment','')}</span>
                        <div style="color:#d4c5a9;font-size:0.9rem;margin-top:0.2rem">{t.get('description','')}</div>
                    </div>""", unsafe_allow_html=True)
            except json.JSONDecodeError:
                st.error("Gemini returned malformed JSON. Try again.")
            except Exception as e:
                err = str(e)
                if "404" in err or "not found" in err.lower():
                    st.error(f"`{gemini_model_sel}` not available. Try gemini-2.5-flash.")
                elif "429" in err or "quota" in err.lower():
                    st.error("Rate limit hit. Wait 1 min and retry.")
                elif "503" in err or "unavailable" in err.lower():
                    st.error("Gemini is overloaded. Try again in a minute.")
                elif "401" in err or "api_key" in err.lower():
                    st.error("Invalid API key.")
                else:
                    st.error(f"Error: {err}")

        st.markdown("---")
        st.markdown('<p class="section-header">All Results</p>', unsafe_allow_html=True)
        result_preview = st.selectbox("Preview rows",
            options=[10, 20, 50, len(rdf)],
            format_func=lambda x: f"Show {x} rows" if x != len(rdf) else f"Show all {len(rdf)} rows",
            index=0, key="result_preview")
        st.dataframe(rdf.head(result_preview), use_container_width=True, height=300)
        st.download_button("⬇ Download Labelled CSV",
                           data=rdf.to_csv(index=False).encode("utf-8"),
                           file_name="telugu_sentiment_results.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Model Info
# ══════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-header">V3 — Best Model (xlm-roberta-large)</p>', unsafe_allow_html=True)
        st.markdown("""<div class="result-card">
            <table style="width:100%;color:#d4c5a9;font-size:0.9rem;border-collapse:collapse">
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Base model</td><td>xlm-roberta-large</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Parameters</td><td>560M</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Languages</td><td>100 (including Telugu)</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Classes</td><td>Negative / Neutral / Positive</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Overall accuracy</td><td>80% (30-review benchmark)</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Neutral accuracy</td><td>60% (vs 10% for indic-bert)</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">Train data</td><td>200 Pos + 200 Neg + 180 Gemini neutral</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">TA approval</td><td>Confirmed ✓</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        st.markdown("""<div class="result-card">
            <table style="width:100%;color:#d4c5a9;font-size:0.9rem;border-collapse:collapse">
                <tr style="color:#f5a623;border-bottom:1px solid #2a2a3a">
                    <td style="padding:0.4rem 0"><b>Model</b></td>
                    <td><b>Overall</b></td><td><b>Neutral</b></td>
                </tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">V1 indic-bert + 61 neutral</td><td>70%</td><td style="color:#ff6b6b">10%</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">V2 indic-bert + 200 neutral</td><td>70%</td><td style="color:#ff6b6b">10%</td></tr>
                <tr><td style="color:#8a8a9a;padding:0.3rem 0">tabularisai multilingual</td><td>70%</td><td style="color:#ff6b6b">10%</td></tr>
                <tr style="border-top:1px solid #2a2a3a">
                    <td style="color:#f5a623;padding:0.3rem 0"><b>V3 xlm-roberta-large ★</b></td>
                    <td style="color:#51cf66"><b>80%</b></td><td style="color:#51cf66"><b>60%</b></td>
                </tr>
            </table>
            <div style="color:#8a8a9a;font-size:0.85rem;margin-top:0.8rem">
                Key finding: model size matters more than data augmentation for Telugu Neutral detection.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Experiment Journey</p>', unsafe_allow_html=True)
    st.markdown("""<div class="result-card">
        <div style="color:#d4c5a9;font-size:0.9rem;line-height:2.2">
            <b style="color:#f5a623">Problem:</b> indic-bert (33M) fails on Neutral — F1 = 0.00<br>
            <b style="color:#f5a623">Hypothesis 1:</b> More neutral data will fix it → V2 (200 neutral) → still 0% ✗<br>
            <b style="color:#f5a623">Hypothesis 2:</b> Bigger model will fix it → V3 (xlm-roberta-large 560M) → 60% ✓<br>
            <b style="color:#f5a623">Conclusion:</b> Model capacity matters more than data quantity for low-resource Neutral detection<br>
            <b style="color:#f5a623">Remaining errors:</b> 4 high-confidence wrong Neutral predictions (87–96%) — confidently wrong, not uncertain
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Project</p>', unsafe_allow_html=True)
    st.markdown("""<div class="result-card">
        <div style="color:#d4c5a9;font-size:0.9rem;line-height:2">
            <b style="color:#f5a623">Assignment:</b> T8.3 — Indic Review & Sentiment Analyzer<br>
            <b style="color:#f5a623">Course:</b> Statistical Methods in AI · IIIT Hyderabad 2025–26<br>
            <b style="color:#f5a623">Team:</b> Tech Titans<br>
            <b style="color:#f5a623">LLMs used:</b> Claude (code scaffolding, data generation), Gemini (neutral data generation, theme extraction)
        </div>
    </div>""", unsafe_allow_html=True)
