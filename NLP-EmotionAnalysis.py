# -*- coding: utf-8 -*-
"""

Author: Ariel Capps
NetId: amc150430

"""

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import re

# Training json is read and transposed so 'body','subreddit', etc. are columns instead of rows
posts = pd.read_json('nlp_train.json')
posts = posts.transpose()


# EMOTIONS:
# anger, anticipation, disgust, fear, joy, love
# optimism, pessimism, sadness, surprise, trust, neutral
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral']

# Declaration of stopwords to be removed and symbols to be kept
stopwords = set(stopwords.words('english'))
symbols = re.compile('[^0-9a-zA-Z_+]')

# Used to clean 'body' by removing stopwords and symbols
def clean(posts):
    posts = posts.lower()

    posts = symbols.sub(' ', posts)
    posts = ' '.join(word for word in posts.split() if word not in stopwords)
    return posts

posts['body'] = posts['body'].apply(clean)

# Parses each post and set count of emotion for that specific post
from collections import defaultdict
y= defaultdict(list)

for i, post in enumerate(posts['emotion']):
    for currEmotion in post:
        if posts['emotion'][i][currEmotion]:
            y[currEmotion].append(1)
        else:
            y[currEmotion].append(0)
labels = pd.DataFrame(y)

# Used to print count of each emotion
# labels.sum(axis=0)


### LSTM MODEL ###

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping

# Tokenizes input and pads sequences
tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(posts['body'].values)

word_index = tokenizer.word_index

x = tokenizer.texts_to_sequences(posts['body'].values)
x = pad_sequences(x, maxlen=100)
y = labels.values

# Split of training data into test/training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

# Creation of LSTM model - parameters were varied multiple times
LSTM_model = Sequential()
LSTM_model.add(Embedding(50000, 100, input_length = x.shape[1]))
LSTM_model.add(SpatialDropout1D(.2))
LSTM_model.add(LSTM(200, dropout=.1, recurrent_dropout=.1))
LSTM_model.add(Dense(12, activation='sigmoid'))
LSTM_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# Training and validation of model is shown along with loss and accuracy per epoch
history = LSTM_model.fit(x_train, y_train, epochs=18, batch_size=64,validation_split=0.15,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Accuracy of model is calculated and printed
accuracy = LSTM_model.evaluate(x_test, y_test)
print('Accuracy: ', accuracy)

# Test data read in
testData = pd.read_json('nlp_test.json')
testData = testData.transpose()

# Test data cleaned and model used for prediction
testData['body'] = testData['body'].apply(clean)
test = tokenizer.texts_to_sequences(testData['body'].values)
test = pad_sequences(x, maxlen=100)
predictions = LSTM_model.predict(test)

        
