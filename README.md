# 📰 Fake News Detection – Streamlit App

A **professional, end-to-end Fake News Detection web application** built using **Python, Streamlit, and Machine Learning**.
The system uses **TF-IDF Vectorization with Logistic Regression** to classify news text as **Fake or Real**, with strong support for **English and Gujarati (including code-mixed text)**.

This project is fully **deployed on Streamlit Cloud** and ready for **resume, internship, and academic submissions**.

---

## 🚀 Live Demo

🔗 **[https://fake-news-detection-app-kkvtwlttag2ytqgghfsc.streamlit.app](https://fake-news-detection-app-kkvtwlttag2ytqgghfsc.streamlit.app)**

---

## ✨ Key Features

* 🔍 Detects **Fake vs Real News** headlines
* 🌐 Supports **English & Gujarati** (code-mixed text)
* 📊 Uses **TF-IDF + Logistic Regression** model
* 🧹 Text preprocessing (stopwords removal, stemming)
* 📂 CSV upload for bulk prediction / training
* ⚡ Interactive and responsive **Streamlit UI**
* ☁️ Deployed on **Streamlit Community Cloud**

---

## 🛠️ Tech Stack

* **Programming Language:** Python 3
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn
* **NLP:** TF-IDF Vectorizer, NLTK
* **Model:** Logistic Regression
* **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

```bash
fake-news-detection-streamlit/
│── app.py                # Main Streamlit application
│── requirements.txt      # Python dependencies
│── fake_news_app.bat     # Windows run script
│── README.md             # Project documentation
│── .gitignore            # Git ignore rules
```

---

## ⚙️ Local Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/jayp7585/fake-news-detection-streamlit.git
cd fake-news-detection-streamlit
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application locally

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 🧪 How the System Works

1. User uploads a CSV file or provides news headlines
2. Text is cleaned and preprocessed
3. **TF-IDF** converts text into numerical features
4. **Logistic Regression** predicts the label
5. Output is shown as:

   * ✅ Real News
   * ❌ Fake News

---

## 📈 Machine Learning Details

* **Vectorizer:** TF-IDF
* **Classifier:** Logistic Regression
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
* **Languages Supported:** English & Gujarati

---

## ☁️ Deployment on Streamlit Cloud (Step-by-Step)

### 1️⃣ Push project to GitHub

Ensure these files exist in the repository:

* `app.py`
* `requirements.txt`
* `README.md`

### 2️⃣ Open Streamlit Cloud

Go to:
👉 [https://streamlit.io/cloud](https://streamlit.io/cloud)

Sign in using **GitHub**.

### 3️⃣ Create a new app

* Repository: `jayp7585/fake-news-detection-streamlit`
* Branch: `main`
* Main file path: `app.py`

Click **Deploy**.

### 4️⃣ Access live app

After deployment, Streamlit provides a public URL:

```
https://fake-news-detection-app-kkvtwlttag2ytqgghfsc.streamlit.app
```

---

## 🧠 Use Cases

* Fake news awareness tools
* Academic & college projects
* NLP and Machine Learning practice
* Internship & resume showcase
* Media credibility analysis

---

## 📸 Screenshots

*Add screenshots of the UI here to make the project more visually appealing.*

---

## 👤 Author

**Jay Panchal**

* GitHub: [https://github.com/jayp7585](https://github.com/jayp7585)
* LinkedIn: [https://www.linkedin.com/in/jay-panchal0324](https://www.linkedin.com/in/jay-panchal0324)

---

## 📜 License

This project is open-source and intended for **educational and learning purposes**.

---

⭐ If you find this project useful, please consider giving it a **star** on GitHub!
