import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
from data_preprocessing import preprocess_data


documents, words, classes = preprocess_data('data/intents.json')


train_x = [] 
train_y = []  
label_encoder = LabelEncoder()
label_encoder.fit(classes)
for doc in documents:
    
    bag_of_words = [0] * len(words)
    pattern_words = doc[0]
    for word in pattern_words:
        if word in words:
            bag_of_words[words.index(word)] = 1 
    train_x.append(bag_of_words)
    label_index = label_encoder.transform([doc[1]])[0]
    train_y.append(to_categorical(label_index, num_classes=len(classes)))
train_x = np.array(train_x)
train_y = np.array(train_y)

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model and data
model.save('models/chatbot_model.h5')
