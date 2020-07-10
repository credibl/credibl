## Trying to use GloVe embeddings and neural networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import keras
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from datahandle import get_html, add_data, clean_text

''' 
Takes in the data
Some cool data sets we could use: https://github.com/KaiDMML/FakeNewsNet
''' 
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')
true_data['is_fake'] = 0
fake_data['is_fake'] = 1

data = pd.concat([true_data, fake_data])
del data['subject']
del data['date']

data = data.sample(frac=1).reset_index(drop=True)

data['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['is_fake'])

# tokenization
NUM_WORDS = 10000
MAX_LEN = 100
tokenizer = text.Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(X_train)

tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=MAX_LEN)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=MAX_LEN)


# read glove word vectors into dictionary
GLOVE_TWITTER = 'glove.twitter.27B/glove.twitter.27B.50d.txt'
def get_kv_pair(line):
    tokens = line.split(' ')
    return tokens[0], np.asarray(tokens[1:], dtype='float32')
embs_dict = dict([get_kv_pair(line) for line in open(GLOVE_TWITTER)])


# generate random weights
all_embs = np.stack(embs_dict.values())
embs_mean, embs_std = all_embs.mean(), all_embs.std()
embs_size = all_embs.shape[1]
word_index = tokenizer.word_index
num_words = min(len(word_index), NUM_WORDS)
embs_matrix = np.random.normal(embs_mean, embs_std, (num_words, embs_size))


# creates an rnn
model = Sequential()
model.add(Embedding(input_dim=NUM_WORDS, output_dim=embs_size, input_length=MAX_LEN, trainable=False, weights = [embs_matrix]))
model.add(LSTM(units = 64, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
model.add(LSTM(units=32, return_sequences=True, recurrent_dropout = 0.1, dropout=0.1))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=256, validation_data=(X_test, y_test), epochs=10, 
                    callbacks=[ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)])