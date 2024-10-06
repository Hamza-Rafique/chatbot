import json
import nltk
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

if __name__ == "__main__":
    documents, words, classes = preprocess_data('data/intents.json')
    print(f"Documents: {documents[:5]}")  # Print sample documents
    print(f"Words: {words[:5]}")          # Print sample words
    print(f"Classes: {classes}")          # Print classes
