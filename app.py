import io
import pickle
import string
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---- Setup ----
try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words_en = set(stopwords.words("english"))

# ---- Gujarati Stopwords (Custom List) ----
gujarati_stopwords = {
    "અને", "છે", "આ", "તે", "ના", "ની", "થી", "પર", "માં", "પણ", "હતું",
    "હતી", "હો", "કર્યું", "કે", "તો", "શું", "કેવી રીતે", "કારણ", "સાથે",
    "બધા", "તેમ", "અમે", "આપણે", "હવે", "જે", "હતો", "હતી", "હશે", "કે", 
    "જોઈએ", "માટે", "છો", "હું", "તું", "તમે", "તેને", "તેણે", "તેના"
}

# Gujarati punctuation/symbols cleanup pattern
GUJ_PUNCTUATION = "ંઃઁ૱૰૳૴૵૶૷૸ૹૺૻૼ૽૾૿"

# ---- Helper: Transliteration Normalization ----
def normalize_transliteration(text: str) -> str:
    """
    Normalize transliterated Gujarati-English words and Surat-related names.
    """
    translit_map = {
        "ahmdabad": "ahmedabad",
        "amdavad": "ahmedabad",
        "srt": "surat",
        "સુરત": "surat",
        "અમદાવાદ": "ahmedabad",
        "ગાંધીનગર": "gandhinagar",
        "વડોદરા": "vadodara",
        "રાજકોટ": "rajkot",
        "ભરૂચ": "bharuch",
        "વાપી": "vapi",
    }
    for k, v in translit_map.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text


# ---- Preprocessing ----
def preprocess_text(s: str) -> str:
    """Clean Gujarati–English mixed text, remove noise, stopwords, and stem English."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()

    # Remove Gujarati punctuation and symbols
    s = re.sub(f"[{re.escape(GUJ_PUNCTUATION)}]", "", s)

    # Remove URLs, numbers, emojis, etc.
    s = re.sub(r"http\S+|www\S+|[\d]|[\U0001F600-\U0001F64F]", "", s)

    # Normalize transliteration (Gujarati ↔ English)
    s = normalize_transliteration(s)

    # Remove English punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = s.split()

    # Remove both Gujarati + English stopwords
    tokens = [t for t in tokens if t not in stop_words_en and t not in gujarati_stopwords]

    # Stem English words (Gujarati left intact)
    stems = [stemmer.stem(t) if re.match(r"[a-z]+", t) else t for t in tokens]

    return " ".join(stems)


# ---- Load pre-trained model ----
MODEL_PATH = "fake_news_tfidf_logreg_surat.pkl"
pretrained = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        pretrained = pickle.load(f)
    st.sidebar.success("✅ Surat Edition Pre-trained Model Loaded.")


# ---- Streamlit UI ----
st.set_page_config(page_title="Surat Fake News Detector", layout="wide")
st.title("📰 Surat Edition Fake News Headline Classifier")
st.markdown("""
Classifies **Times of India (Surat)** and **Sandesh (Surat)** headlines as *Real* or *Fake*.
Supports **Gujarati–English code-mixing**, **transliteration**, and **local entity normalization**.
""")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with columns like [headline/text/title, label]", type=["csv"]
)
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.5, 0.2)


# ---- Training function ----
def train_and_evaluate(df: pd.DataFrame):
    df = df.copy()

    possible_text_cols = ["headline", "title", "text"]
    text_col = None
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        st.error(f"❌ Expected one of columns: {possible_text_cols}")
        st.stop()

    if "label" not in df.columns:
        st.error("❌ Missing required 'label' column.")
        st.stop()

    st.info(f"Using text column: **{text_col}**")

    df["clean_text"] = df[text_col].astype(str).apply(preprocess_text)
    y_raw = df["label"].astype(str).str.lower()

    # Convert label to numeric
    if set(y_raw.unique()) <= {"0", "1"}:
        y = y_raw.astype(int).values
    else:
        y = y_raw.apply(lambda x: 1 if "real" in x else 0).values

    # TF-IDF with 1–3 grams for mixed-script robustness
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 3), analyzer="word")
    X = vec.fit_transform(df["clean_text"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = LogisticRegression(C=1.2, solver="liblinear", max_iter=1200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "vectorizer": vec,
        "model": model,
    }
    return metrics


# ---- App Logic ----
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("🚀 Train Surat Edition Model"):
        metrics = train_and_evaluate(df)
        st.success("✅ Model trained successfully on Surat dataset!")
        st.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
        st.write(f"**Precision:** {metrics['precision']:.3f}")
        st.write(f"**Recall:** {metrics['recall']:.3f}")
        st.write(f"**F1 Score:** {metrics['f1']:.3f}")

        buffer = io.BytesIO()
        pickle.dump({"vectorizer": metrics["vectorizer"], "model": metrics["model"]}, buffer)
        buffer.seek(0)
        st.download_button("💾 Download Surat Model", buffer, "fake_news_tfidf_logreg_surat.pkl")

elif pretrained:
    st.subheader("🤖 Using Pre-trained Surat Model")
    user_headline = st.text_area("Enter a Gujarati–English mixed headline:", height=120)
    if st.button("🔍 Predict"):
        if user_headline.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            vec, model = pretrained["vectorizer"], pretrained["model"]
            clean = preprocess_text(user_headline)
            X_user = vec.transform([clean])
            prob = model.predict_proba(X_user)[0][1]
            label = "Real" if prob >= 0.5 else "Fake"
            if label == "Real":
                st.success(f"✅ This headline appears **REAL** (prob={prob:.3f})")
            else:
                st.error(f"🚨 This headline appears **FAKE** (prob={prob:.3f})")
else:
    st.info("Upload Surat dataset to train or use pre-trained model.")
