## Trying to use GloVe embeddings and neural networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout

from sklearn.model_selection import train_test_split

true_data = pd.read_csv('True.csv')
false_data = pd.read_csv('False.csv')

data = pd.concat([true_data, false_data])
del data.subject
del data.date

print(data[['text']].head())