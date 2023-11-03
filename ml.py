from jinja2 import Undefined
import spacy
from spacy.lang.fr.examples import sentences
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deep_translator import GoogleTranslator

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv("data/train.csv")

def preprocess_text(text):
    return text

data["review"] = data["review"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data["review"], data["sentiment"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

X_train_vectors = vectorizer.fit_transform(X_train)
classifier.fit(X_train_vectors, y_train)

def predict_sentiment(comment):

    comment = GoogleTranslator(source='auto', target='en').translate(comment)

    comment = preprocess_text(comment)
    comment_vector = vectorizer.transform([comment])
    prediction = classifier.predict(comment_vector)
    prediction_proba = classifier.predict_proba(comment_vector)
    
    if prediction_proba.max() < 0.6:
        return 'neutral'
    else:
        return prediction[0]

X_test_vectors = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy * 100:.2f}%")

def moderate_comment(comment):
    sentiment = predict_sentiment(comment)

    if sentiment == "negative":
        return 0
    elif sentiment == "positive":
        return 1
    else:
        return 2