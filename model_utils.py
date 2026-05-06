"""
model_utils.py - Train and manage spam detection models
Uses only scikit-learn compatible models for Streamlit Cloud
"""

import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib
import os

# Download NLTK data
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


# ─── Built-in training data ───────────────────────────────────────────────────

SPAM_EMAILS = [
    "Congratulations! You've won a $1,000 gift card. Click here to claim your prize now!",
    "FREE MONEY!!! Make $5000 per week working from home. No experience needed!",
    "URGENT: Your account has been compromised. Verify your details immediately!",
    "You have been selected for a special offer. Act now before it expires!",
    "Buy cheap Viagra online. Best prices guaranteed. No prescription needed!",
    "Nigerian prince needs your help. $10 million transfer. 30% commission for you!",
    "WINNER WINNER! You are the lucky winner of our weekly lottery. Claim now!",
    "Earn money fast! Join our MLM network and make thousands every month!",
    "FINAL NOTICE: Your loan application approved. Get $50,000 instantly!",
    "Hot singles in your area want to meet you. Click here to find them!",
    "Lose 30 pounds in 30 days with this miracle pill! Doctors HATE this!",
    "Your PayPal account has been suspended. Verify immediately to restore access!",
    "Get rich quick! Invest $100 today and earn $10,000 tomorrow!",
    "Congratulations! Apple iPhone 15 Pro giveaway! You are today's winner!",
    "CLICK HERE: Exclusive deal for you only. 99% off everything store-wide!",
    "Make money online from home. $500/day guaranteed. Start today!",
    "Special offer: Free casino chips! Sign up now and get 1000 free spins!",
    "Your computer has a virus! Call this number immediately: 1-800-SCAM!",
    "Refinance your mortgage now! Lowest rates ever! Call us immediately!",
    "You've been pre-approved for a credit card with no credit check!",
    "Exclusive: Weight loss secret revealed! Celebrities don't want you to know this!",
    "ATTENTION: IRS final notice. Pay now or face immediate arrest!",
    "Meet hot local women tonight. No strings attached. Free membership now!",
    "Double your Bitcoin in 24 hours! Guaranteed returns! Limited time offer!",
    "FREE iPhone! Just complete this survey and provide your credit card details!",
    "You have unclaimed rewards worth $500. Redeem now before they expire!",
    "Urgent: Your social security number has been suspended. Call immediately!",
    "Buy prescription drugs online without prescription. Cheap and fast delivery!",
    "Amazing work from home opportunity! $2000/day guaranteed with zero effort!",
    "Your email won the Microsoft lottery! Claim your $1 million prize now!",
    "Special discount on all products. Use code SPAM50 for 50% off everything!",
    "LIMITED TIME: Penny stocks about to explode! Invest now for 10000% returns!",
    "WARNING: Your bank account will be closed. Verify your information immediately!",
    "Cheap designer watches, bags, shoes. Authentic replicas at lowest prices!",
    "You qualify for a student loan forgiveness program. Apply immediately!",
    "BREAKING: Secret investment opportunity. Guaranteed 200% return in 7 days!",
    "FREE vacation to Cancun! You've been selected. Claim your trip today!",
    "Increase your male performance! Doctors recommend this secret supplement!",
    "Hack any WiFi password! Download our free tool now. Works 100%!",
    "You owe back taxes. Pay immediately or face criminal prosecution!",
]

HAM_EMAILS = [
    "Hi John, could you please send me the report for the Q3 analysis? Thanks.",
    "Meeting tomorrow at 3pm in conference room B. Please bring your laptop.",
    "Your package has been shipped and will arrive by Thursday.",
    "Happy birthday! Hope you have a wonderful day with friends and family.",
    "The project deadline has been moved to next Friday. Please update your schedule.",
    "Can you pick up milk and eggs on your way home from work today?",
    "Your doctor's appointment is confirmed for Monday at 10:30 AM.",
    "Thank you for your application. We will review it and get back to you shortly.",
    "The team meeting has been rescheduled to 2pm due to a conflict.",
    "Please review the attached document and provide your feedback by end of day.",
    "Reminder: Your subscription renewal is due on the 15th of this month.",
    "The quarterly report is ready for your review. Please find it attached.",
    "We wanted to let you know your order has been successfully delivered.",
    "Hi Mom, just wanted to check in. Hope you're doing well! Call me later.",
    "The conference registration is now open. Early bird discounts end Friday.",
    "Your flight booking is confirmed. Check-in opens 24 hours before departure.",
    "Please complete the employee satisfaction survey by end of this week.",
    "The library books you requested are now available for pickup.",
    "Friendly reminder about tomorrow's parent-teacher conference at 6pm.",
    "Your electricity bill for this month is $87.50. Due date: March 15.",
    "The software update is complete. Please restart your computer at your convenience.",
    "Welcome to the team! Your first day orientation begins at 9am Monday.",
    "Can we reschedule our lunch meeting? Something urgent came up at work.",
    "The annual company picnic is this Saturday. Families are welcome!",
    "Your refund of $45.00 has been processed and will appear within 3-5 business days.",
    "Study group meeting tonight at 7pm in the library. Chapter 5 and 6 review.",
    "Please submit your timesheet by noon on Friday for payroll processing.",
    "The new version of the software includes improved performance and bug fixes.",
    "Thank you for attending the workshop. The slides are available for download.",
    "Your reservation at The Grand Hotel is confirmed for July 14-17.",
    "The campus library will be closed on Monday for maintenance.",
    "Results of the quarterly performance review are now available in HR portal.",
    "Please find the minutes from last week's board meeting attached.",
    "The neighborhood association meeting is on Wednesday at 7:30pm.",
    "Your tax documents are ready. Please log in to access them.",
    "Reminder: Annual dental cleaning appointment tomorrow at 2pm.",
    "The research paper submission deadline has been extended by one week.",
    "Congratulations on your promotion! Well deserved recognition for your hard work.",
    "The office will be closed on Friday for the public holiday.",
    "Please update your emergency contact information in the HR system.",
    "Your car service appointment is scheduled for Saturday at 8am.",
    "The internship applications are now being reviewed. Decisions by end of month.",
    "Team lunch this Friday. We're going to that Italian place on Main Street.",
    "The client presentation went well. They loved our proposal!",
    "Your grades for the fall semester are now available on the student portal.",
]


def preprocess_text(text):
    """Clean and preprocess email text."""
    download_nltk_data()

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\$[\d,]+', ' money ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    stemmer = PorterStemmer()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def get_spam_features(text):
    """Extract spam indicator features from raw text."""
    features = {}
    text_lower = text.lower()

    spam_keywords = [
        'free', 'winner', 'won', 'prize', 'claim', 'urgent', 'congratulations',
        'guaranteed', 'million', 'cash', 'earn', 'money', 'credit', 'buy',
        'cheap', 'offer', 'deal', 'click', 'limited', 'exclusive', 'special',
        'discount', 'percent off', 'risk free', 'act now', 'order now',
        'apply now', 'call now', 'subscribe', 'cancel', 'password', 'account',
        'verify', 'suspended', 'lottery', 'selected', 'pre-approved',
    ]

    features['spam_keyword_count'] = sum(1 for kw in spam_keywords if kw in text_lower)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text_lower))
    features['money_mentions'] = len(re.findall(r'\$[\d,]+|free money|earn money', text_lower))
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0

    return features


def build_and_train_model():
    """Build dataset and train models. Returns trained pipeline + metrics."""
    texts = SPAM_EMAILS + HAM_EMAILS
    labels = [1] * len(SPAM_EMAILS) + [0] * len(HAM_EMAILS)

    processed = [preprocess_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Individual pipelines
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ('clf', MultinomialNB(alpha=0.1)),
    ])

    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])

    rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    models = {
        'Naive Bayes': nb_pipeline,
        'Logistic Regression': lr_pipeline,
        'Random Forest': rf_pipeline,
    }

    metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

    # Best model = Logistic Regression (default)
    best_model = lr_pipeline

    return models, best_model, metrics, X_test, y_test


def predict_email(text, model):
    """Predict spam/ham for a single email. Returns label, confidence, features."""
    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]
    confidence = float(proba[prediction])
    features = get_spam_features(text)

    return {
        'prediction': int(prediction),
        'label': 'SPAM' if prediction == 1 else 'NOT SPAM',
        'confidence': confidence,
        'spam_probability': float(proba[1]),
        'ham_probability': float(proba[0]),
        'features': features,
    }
