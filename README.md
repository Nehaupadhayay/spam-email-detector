# 🛡️ SpamShield AI — Email Spam Detector
### Final Year Project | Machine Learning + NLP + Streamlit

---

## 📌 Overview

SpamShield AI is a machine learning-powered email spam detection web application built for a Final Year Project. It uses NLP preprocessing and three ML classifiers to accurately identify spam emails in real-time. The app is fully deployable on **Streamlit Cloud** with zero infrastructure setup.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-time Detection** | Paste any email text and instantly detect spam/ham |
| 📊 **Probability Score** | Gauge chart showing exact spam probability |
| 🤖 **3 ML Models** | Switch between Naive Bayes, Logistic Regression, Random Forest |
| 🎚️ **Threshold Control** | Adjust detection sensitivity in the sidebar |
| ⚡ **Live Stats** | Word count, CAPS ratio, URL count, live risk estimate |
| 🔬 **Indicator Pills** | Visual spam feature analysis (keywords, exclamations, etc.) |
| 📋 **Batch Analysis** | Analyze multiple emails at once |
| 📈 **Analytics Dashboard** | Model comparison charts, radar chart, confusion matrix |

---

## 🚀 Deploying on Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: SpamShield AI"
git remote add origin https://github.com/YOUR_USERNAME/email_spam_detector.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

That's it — no API keys or secrets needed!

---

## 💻 Running Locally

```bash
# Clone or open the folder in VS Code
cd email_spam_detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
email_spam_detector/
│
├── app.py              # Main Streamlit application (UI + logic)
├── model_utils.py      # ML models, training data, preprocessing, prediction
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages (for Streamlit Cloud)
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit theme and server config
```

---

## 🧠 ML Pipeline

```
Raw Email Text
     ↓
Lowercase + URL/Email Tagging
     ↓
Punctuation Removal + Number Normalization
     ↓
Stopword Removal (NLTK)
     ↓
Porter Stemming
     ↓
TF-IDF Vectorization (bigrams, 3000 features)
     ↓
ML Classifier (NB / LR / RF)
     ↓
Spam Probability + Label
```

---

## 🤖 Models

| Model | Description |
|---|---|
| **Naive Bayes** | Probabilistic text classifier using word frequency |
| **Logistic Regression** | Linear binary classifier, often best for text |
| **Random Forest** | Ensemble of decision trees for robust classification |

---

## 📦 Dependencies

- `streamlit` — Web application framework
- `scikit-learn` — ML models and TF-IDF vectorizer
- `nltk` — Natural language processing (stopwords, stemming)
- `plotly` — Interactive charts and visualizations
- `pandas` / `numpy` — Data manipulation
- `joblib` — Model serialization

---

## 👨‍🎓 Project Info

- **Type**: Final Year Capstone Project
- **Domain**: Machine Learning / Natural Language Processing
- **Deployment**: Streamlit Cloud
- **Language**: Python 3.10+

---

> Built with ❤️ using Python & Streamlit
