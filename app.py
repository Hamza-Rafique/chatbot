from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json


app = Flask(__name__)
CORS(app) 

lemmatizer = WordNetLemmatizer()

model = load_model('models/chatbot_model.h5')

# Load intents
with open('data/intents.json', 'r') as f:
    intents = json.load(f)

# Load tokenizer
with open('data/tokenizer.json', 'r') as f:
    tokenizer_data = f.read()
    tokenizer = tokenizer_from_json(tokenizer_data)

# Load label encoder
with open('data/label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def predict_class(sentence):
    """Predicts the class for the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    seq = tokenizer.texts_to_sequences([sentence_words])
    padded_seq = pad_sequences(seq, maxlen=20, padding='post')

    prediction = model.predict(padded_seq, verbose=0)
    predicted_class_index = np.argmax(prediction)

    if isinstance(label_encoder_data, list):
        intent_tag = label_encoder_data[predicted_class_index]
    return intent_tag

def get_response(intent_tag):
    """Fetches the response for the predicted intent."""
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the POST request for chatbot communication."""
    user_input = request.json.get("message")
    if user_input:
        intent_tag = predict_class(user_input)
        response = get_response(intent_tag)
        return jsonify({"response": response})
    else:
        return jsonify({"error": "No message provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
