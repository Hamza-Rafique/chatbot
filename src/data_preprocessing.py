import json
import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_data(intents_file):
    with open(intents_file) as file:
        data = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']

    for intent in data['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    return documents, words, classes
