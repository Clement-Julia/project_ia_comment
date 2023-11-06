import pandas as pd
import numpy as np
import spacy
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model
from deep_translator import GoogleTranslator

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer()
variables = ["toxic", "obscene", "insult", "identity_hate"]

def preprocess_text(text):
    text = re.sub(r"<.*?>", "", text)
    doc = nlp(text, disable=["parser", "tagger", "ner", "textcat"])
    simplified_text = " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop and not token.is_digit])
    
    return simplified_text

def clear_data(data, var):
    data_copy = data.copy()

    data_copy["comment_text"] = data_copy["comment_text"].astype(str)
    data_copy = data_copy.drop_duplicates(subset="comment_text")
    data_copy["comment_text"] = data_copy["comment_text"].dropna()

    toxic_samples = data_copy[data_copy[var] == 1].drop_duplicates(subset="comment_text")
    non_toxic_samples = data_copy[data_copy[var] == 0].drop_duplicates(subset="comment_text")
    num_toxic_samples = min(50000, len(toxic_samples))
    toxic_samples = toxic_samples.sample(n=num_toxic_samples, random_state=42)
    non_toxic_samples = non_toxic_samples.sample(n=num_toxic_samples, random_state=42)
    new_data = pd.concat([toxic_samples, non_toxic_samples])

    data = new_data.sample(frac=1, random_state=42)
    data["comment_text"] = data["comment_text"].apply(preprocess_text)

    return data

def prediction_toxicity(model, comment):
    max_sequence_length = 100
    comment_sequence = tokenizer.texts_to_sequences([comment])
    comment_padded = pad_sequences(comment_sequence, maxlen=max_sequence_length, padding='post')

    predictions = model.predict(comment_padded)
    predicted_class = np.argmax(predictions)

    return predicted_class, predictions

def get_toxicity(comment):
    toxic_score = 0

    for toxic_var in variables:
        model = load_model(f"static/model/{toxic_var}.keras")
        comment = GoogleTranslator(source='auto', target='en').translate(comment)
        comment = preprocess_text(comment)

        predicted_class = prediction_toxicity(model, comment)
        toxic_score = toxic_score + predicted_class
    
    return toxic_score