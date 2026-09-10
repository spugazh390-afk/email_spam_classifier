# 🛡️ SpamGuard AI - Email Spam Classification System

An advanced, production-grade Email Spam Classification web application built with **Python (Flask, Scikit-Learn)**, **SQLite**, and a modern dashboard UI (**Tailwind CSS + Chart.js**).

---

## 🌟 4 Key Features Implemented

### 1. 📊 Confidence Score
- **Probabilistic Scoring**: Calculates precise class likelihoods via Scikit-Learn's `predict_proba()`.
- **Dynamic Risk Categorization**:
  - 🔴 **High Risk Spam** (&ge; 80% spam probability)
  - 🟡 **Suspicious** (50% &ndash; 79% spam probability)
  - 🟢 **Safe / Legitimate** (< 50% spam probability)
- **Dual Visual Meter**: Visual progress bar simultaneously displaying Spam % vs Safe % balance.
- **Adjustable Decision Threshold**: Slider in the UI allows fine-tuning the spam trigger sensitivity (10% &ndash; 90%).

### 2. 🧠 Explainable AI (XAI)
- **Linear Feature Attribution**: Uses model TF-IDF $\times$ coefficient weights to calculate exact token impact scores.
- **Interactive Visual Highlighter**:
  - Highlights spam-triggering words in **soft red** with weights (e.g. `urgent (+0.82)`, `lottery (+1.10)`).
  - Highlights legitimate business/communication words in **soft green** (e.g. `meeting (-0.95)`, `agenda (-0.72)`).
  - Hover tooltips reveal exact model coefficients.
- **Top Contributing Factors Breakdown**: Ranked lists of top spam-inducing and ham-inducing words.
- **Natural Language Verdict**: Human-readable narrative explaining *why* the AI made this decision.

### 3. 📜 Prediction History
- **Persistent SQLite Audit Trail** (`spam_history.db`):
  - Tracks ID, timestamp, subject, snippet, prediction, confidence, spam/ham probabilities, and trigger keywords.
- **Live Search & Filters**: Search across email subjects/content and filter by Spam or Safe.
- **Inspection Modal**: Re-open any past email to view full XAI token highlights and AI explanation.
- **CSV Data Export**: One-click download of all prediction audit logs.

### 4. 🎛️ Admin Dashboard
- **Key Performance Indicators (KPIs)**:
  - Total Emails Scanned
  - Spam Detected Count & Spam Ratio (%)
  - Legitimate (Ham) Count & Safe Ratio (%)
  - Average Confidence Score (%)
- **Interactive Visual Analytics (Chart.js)**:
  - 🍩 **Class Distribution**: Donut chart comparing Spam vs Ham.
  - 📊 **Top Trigger Keywords**: Bar chart tracking the most frequent spam words detected across history.
  - 📈 **Activity Trend**: Line chart tracking daily classification volume over the last 7 days.
- **Model Architecture Diagnostics**:
  - Displays algorithm, vectorizer settings (Unigrams + Bigrams), active vocabulary size, and calibration status.
- **Admin Controls**: One-click database reset and audit CSV download.

---

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have Python (version 3.10+) installed.

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
```bash
py train_model.py
```
This trains the TF-IDF + Logistic Regression classifier on realistic spam/ham email datasets and generates `model.joblib`.

### 3. Start the Web Server
```bash
py app.py
```

### 4. Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser:
- **Classifier**: `http://127.0.0.1:5000/`
- **Prediction History**: `http://127.0.0.1:5000/history`
- **Admin Dashboard**: `http://127.0.0.1:5000/admin`

---

## 📁 Project Structure

```
email_spam_classifier/
│── app.py                    # Main Flask application & REST API routes
│── model.py                  # SpamClassifier class + XAI Feature Attribution engine
│── train_model.py            # Model training, validation, and evaluation script
│── database.py               # SQLite database setup, history tracking & analytics
│── test_classifier.py        # Automated test suite verifying all 4 features
│── requirements.txt          # Python library dependencies
│── model.joblib              # Serialized trained model pipeline
│── spam_history.db           # SQLite database storing prediction logs
│── static/
│   └── css/
│       └── style.css         # Styling for XAI highlight chips and animations
└── templates/
    ├── base.html             # Common navbar, footer, Tailwind CDN & Chart.js
    ├── index.html            # Classifier page with Confidence Meter & XAI
    ├── history.html          # Prediction History table, search, & detail modal
    └── admin.html            # Admin Analytics Dashboard with Chart.js charts
```

---

##  tamil (தமிழ் விளக்கம்)
- **Confidence Score**: Email எவ்வளவு சதவீதம் Spam அல்லது Safe என்று துல்லியமாக கணித்து Color Bar-ல் காட்டும்.
- **Explainable AI**: Spam-க்கு காரணமான வார்த்தைகளை சிவப்பிலும் (Red highlight), நல்ல வார்த்தைகளை பச்சையிலும் (Green highlight) காட்டி, AI ஏன் இந்த முடிவை எடுத்தது என்று விளக்கமளிக்கும்.
- **Prediction History**: முந்தைய scan செய்த அனைத்து emails-களையும் Database-ல் சேமித்து Search மற்றும் CSV Export செய்ய வழிவகை செய்கிறது.
- **Admin Dashboard**: Total scans, Spam %, Graph charts மற்றும் Top Spam Keywords ஆகியவற்றை எளிதாக கண்காணிக்கலாம்.
