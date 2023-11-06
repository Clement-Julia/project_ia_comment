import os
import pandas as pd
import numpy as np
import string
import spacy
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import load_model
from deep_translator import GoogleTranslator

# Désactiver les avertissements
# warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer()

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

if os.path.exists("toxic.keras"):
    model = load_model("toxic.keras")
else:
    data = pd.read_csv("../data/train_1M.csv")
    data = clear_data(data, "toxic")

    X_train, X_test, y_train, y_test = train_test_split(data["comment_text"], data["toxic"], test_size=0.2, random_state=42)

    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    max_sequence_length = 100
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post')

    model = tf.keras.Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length),
        LSTM(64),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)
    model.fit(X_train_padded, y_train_encoded, epochs=10, batch_size=32, validation_split=0.1)

    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=3)

    model.save("toxic.keras")

def prediction_toxicity(comment):
    max_sequence_length = 100
    comment_sequence = tokenizer.texts_to_sequences([comment])
    comment_padded = pad_sequences(comment_sequence, maxlen=max_sequence_length, padding='post')

    class_labels = [0, 1, 2]
    predictions = model.predict(comment_padded)
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

def get_toxicity(comment):
    comment = GoogleTranslator(source='auto', target='en').translate(comment)
    comment = preprocess_text(comment)

    predicted_class = prediction_toxicity(model, comment)
    return predicted_class