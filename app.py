import streamlit as st
import pickle
import numpy as np

# -------- Load Models --------
lda_model = pickle.load(open("lda_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
topic_names = pickle.load(open("topic_names.pkl", "rb"))

# -------- Page Config --------
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------- Premium CSS --------
st.markdown("""
<style>

/* Hide Streamlit UI */
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}


/* Background */
[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(10,15,25,0.55), rgba(10,15,25,0.55)),
        url("https://images.unsplash.com/photo-1504711434969-e33886168f5c");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}


/* ============================= */
/* PAGE ENTRY ANIMATION */
/* ============================= */

.block-container {
    padding-top: 70px;
    max-width: 750px;

    animation: pageEnter 0.8s ease-out;
}

@keyframes pageEnter {

    from {
        opacity: 0;
        transform: translateY(40px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}



/* ============================= */
/* PROFESSIONAL TITLE */
/* ============================= */

.title {

    font-size: 44px;
    font-weight: 700;

    color: #ffffff;

    text-align: center;

    letter-spacing: 0.5px;

    margin-bottom: 8px;

    animation: fadeSlide 1s ease-out;
}


/* LIVE underline animation */

.title::after {

    content: "";

    display: block;

    width: 80px;

    height: 3px;

    margin: 12px auto 0;

    border-radius: 3px;

    background: linear-gradient(90deg,#ff512f,#dd2476);

    animation: underlineMove 3s ease-in-out infinite;
}

@keyframes underlineMove {

    0% {
        width: 60px;
        opacity: 0.6;
    }

    50% {
        width: 140px;
        opacity: 1;
    }

    100% {
        width: 60px;
        opacity: 0.6;
    }
}


@keyframes fadeSlide {

    from {
        opacity: 0;
        transform: translateY(-40px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}



/* Subtitle */

.subtitle {

    font-size: 18px;

    color: rgba(255,255,255,0.75);

    text-align: center;

    margin-bottom: 35px;

    animation: fadeIn 1.5s ease;
}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}



/* ============================= */
/* GLASS BLACK INPUT */
/* ============================= */

[data-testid="stTextArea"] {
    background: transparent !important;
}

[data-testid="stTextArea"] > div {
    background: transparent !important;
    border: none !important;
}

[data-testid="stTextArea"] > div > div {

    background: rgba(35, 40, 55, 0.55) !important;

    border-radius: 20px !important;

    border: none !important;

    backdrop-filter: blur(25px) !important;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.55) !important;
}

[data-testid="stTextArea"] textarea {

    background: transparent !important;

    color: white !important;

    border: none !important;

    outline: none !important;

    font-size: 16px !important;

    padding: 18px !important;
}

[data-testid="stTextArea"] textarea::placeholder {

    color: rgba(255,255,255,0.5) !important;
}



/* ============================= */
/* BUTTON ANIMATION */
/* ============================= */

.stButton > button {

    width: 100%;

    background: linear-gradient(90deg,#ff512f,#dd2476);

    color: white;

    font-size: 18px;
    font-weight: 600;

    border-radius: 12px;

    padding: 12px;

    border: none;

    transition: all 0.2s ease;
}


/* Hover */

.stButton > button:hover {

    transform: scale(1.04);

    box-shadow: 0 10px 30px rgba(255,80,120,0.4);
}


/* Click animation */

.stButton > button:active {

    transform: scale(0.96);

    box-shadow: 0 5px 15px rgba(0,0,0,0.4);
}



/* ============================= */
/* RESULT APPEAR ANIMATION */
/* ============================= */

.result-box {

    margin-top: 25px;

    background: rgba(255,255,255,0.08);

    border-radius: 15px;

    padding: 18px;

    text-align: center;

    font-size: 20px;

    color: white;

    border: 1px solid rgba(255,255,255,0.15);

    backdrop-filter: blur(15px);

    animation: resultPop 0.5s ease-out;
}

@keyframes resultPop {

    from {
        opacity: 0;
        transform: scale(0.9) translateY(10px);
    }

    to {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)



# -------- Title --------

st.markdown(
    '<div class="title">News Topic Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered topic detection using Latent Dirichlet Allocation</div>',
    unsafe_allow_html=True
)



# -------- Input --------

news = st.text_area(
    "",
    height=160,
    placeholder="Paste your news article here..."
)



# -------- Button --------

if st.button("🚀 Classify Topic"):

    if news.strip() == "":
        st.warning("Please enter news text")

    else:

        vec = vectorizer.transform([news])

        topic_probs = lda_model.transform(vec)

        topic_index = np.argmax(topic_probs)

        topic = topic_names[topic_index]

        st.markdown(
            f'<div class="result-box">📌 Classified Topic: <b>{topic}</b></div>',
            unsafe_allow_html=True
        )
