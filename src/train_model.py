import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Load the intents data
with open('data/intents.json', 'r') as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Tokenizer to convert sentences to sequences
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
X = tokenizer.texts_to_sequences(sentences)
max_length = max(len(sequence) for sequence in X)
X = pad_sequences(X, maxlen=max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = tf.keras.utils.to_categorical(y)

# Define model
embedding_dim = 100
num_classes = len(set(labels))
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Save the model, tokenizer, and label encoder
model.save('models/chatbot_model.h5')

tokenizer_json = tokenizer.to_json()
with open('data/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

with open('data/label_encoder.json', 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)

print("Model training completed and files saved.")
