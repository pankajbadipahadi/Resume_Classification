import streamlit as st
import pickle
import fitz
from docx import Document
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- Load Model ----------------
with open("resume_model.pkl", "rb") as f:
    model = pickle.load(f)

vectorizer = model.named_steps['tfidf'] if hasattr(model, "named_steps") else None

# ---------------- Session State ----------------
if "total_resumes" not in st.session_state:
    st.session_state.total_resumes = 0

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Page Config ----------------
st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")

# ---------------- Theme Toggle ----------------
theme = st.toggle("🌙 Dark Mode", value=True)

bg_color = "#0e1117" if theme else "#f5f7fa"
text_color = "#ffffff" if theme else "#000000"
box_color = "#1c1f26" if theme else "#eaf0ff"

st.markdown(f"""
<style>
body {{ background-color: {bg_color}; color: {text_color}; }}
.result-box {{
    background-color: {box_color};
    padding: 1.2rem;
    border-radius: 12px;
    margin-top: 1.5rem;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center;'>📄 Resume Classifier</h1>", unsafe_allow_html=True)
st.write("Predict job roles from resumes using NLP.")

# ---------------- Input ----------------
mode = st.radio("Choose input:", ["Upload Resume", "Paste Text"], horizontal=True)

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    if ext == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif ext == "docx":
        d = Document(file)
        return " ".join([p.text for p in d.paragraphs])
    return ""

resume_text = ""
if mode == "Upload Resume":
    file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    if file:
        resume_text = extract_text(file)
else:
    resume_text = st.text_area("Paste Resume Text", height=250)

# ---------------- Utility ----------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    if resume_text.strip() == "":
        st.warning("Please provide resume input.")
    else:
        scores = model.decision_function([resume_text])[0]
        probs = softmax(scores)

        best_idx = np.argmax(probs)
        pred_role = model.classes_[best_idx]
        confidence = probs[best_idx] * 100

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        st.write(f"**Job Role:** {pred_role}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Explanation ----------
        if vectorizer:
            tfidf_vec = vectorizer.transform([resume_text])
            feature_names = vectorizer.get_feature_names_out()
            top_words_idx = np.argsort(tfidf_vec.toarray()[0])[-8:]
            keywords = [feature_names[i] for i in top_words_idx[::-1]]

            st.subheader("🧠 Important Keywords")
            st.write(", ".join(keywords))

        # ---------- Update Session History ----------
        st.session_state.total_resumes += 1
        st.session_state.history.append((pred_role, confidence))

        # ---------- Prepare Results ----------
        df_result = pd.DataFrame([[pred_role, confidence]], columns=["Role", "Confidence (%)"])

        # ---------- CSV Download ----------
        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "prediction.csv", "text/csv")

        # ---------- PDF Download ----------
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.drawString(50, 800, "Resume Classification Result")
        c.drawString(50, 760, f"Predicted Role: {pred_role}")
        c.drawString(50, 730, f"Confidence: {confidence:.2f}%")
        c.save()

        st.download_button("⬇ Download PDF", buffer.getvalue(), "prediction.pdf", "application/pdf")

# ---------------- Analytics Dashboard ----------------
st.markdown("---")
st.header("📊 Analytics Dashboard")

history = pd.DataFrame(st.session_state.history, columns=["Role", "Confidence"])

if history.empty:
    st.info("No predictions yet.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", st.session_state.total_resumes)
    col2.metric("Unique Roles", history["Role"].nunique())
    col3.metric("Avg Confidence", f"{history['Confidence'].mean():.2f}%")

    st.subheader("Role Distribution")
    st.bar_chart(history["Role"].value_counts())

    st.subheader("Confidence Trend")
    st.line_chart(history["Confidence"])
