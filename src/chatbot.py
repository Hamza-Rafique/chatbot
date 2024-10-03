import random
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load trained model
model = load_model('models/chatbot_model.h5')

# Load intents
with open('data/intents.json', 'r') as file:
    intents = json.load(file)

# Load tokenizer
with open('data/tokenizer.json', 'r') as f:
    tokenizer_data = f.read()  # Read the JSON file as a string
    tokenizer = tokenizer_from_json(tokenizer_data)  # Load tokenizer from JSON string

# Load label encoder
with open('data/label_encoder.json', 'r') as f:
    label_encoder_data = json.load(f)

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def predict_class(sentence):
    """Tokenizes the input sentence, converts it to sequences, and predicts the class."""
    # Clean up sentence
    sentence_words = clean_up_sentence(sentence)
    
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([sentence_words])
    padded_seq = pad_sequences(seq, maxlen=20, padding='post')  # Adjust `maxlen` to fit your model

    # Predict the intent
    prediction = model.predict(padded_seq)
    predicted_class_index = np.argmax(prediction)
    
    # Map the predicted index back to a label
    intent_tag = list(label_encoder_data.keys())[list(label_encoder_data.values()).index(predicted_class_index)]
    
    return intent_tag

def get_response(intent_tag):
    """Fetches the response for the predicted intent."""
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])

def chatbot_response(msg):
    """Generates a response from the chatbot."""
    intent_tag = predict_class(msg)
    return get_response(intent_tag)

def chat():
    """Starts the chatbot conversation."""
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        response = chatbot_response(msg)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
