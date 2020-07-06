import numpy as np 
import pandas as pd
from sklearn.model_selection import *
from sklearn import linear_model
import matplotlib.pyplot as plt
from collections import Counter
import enchant
import html2text
import requests
import string
from keras.preprocessing import text

dict = enchant.Dict('en_US')


## Input code, nothing special here, but here is the data we used, its from kaggle, check it out if you have time: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv

fake_data = pd.read_csv('Fake.csv')
fake_data['is_fake'] = 1

true_data = pd.read_csv('True.csv')
true_data['is_fake'] = 0

data = fake_data.append(true_data)

del data['subject']
del data['date']

## ATTEMPT # 1 
## 88% ACCURATE
## The code below is us trying to make a decision boundary based on the number of capitalized words in the title of the article.
## Our guess is that articles with a lot of capitalized words are probably very sensationalist and are not that credible

def is_acronym(word):
  return word.count('.') >= len(word) / 2 or word.isupper() and \
      not dict.check(''.join([x for x in word if x.isalpha()]).lower())

def is_caps_word(word):
  for p in string.punctuation.replace('.', ''):
    word = word.replace(p, '')
  return word.isupper() and not is_acronym(word) and len(word) > 2

## returns the capitalized words in the title of an article in a tokenized list
def get_caps_words(title):
  return [x for x in title.split(' ') if is_caps_word(x)]

## returns the number of capitalized words in the title (accounts for acronyms/exceptions)
def get_num_caps_words(title):
  return sum(is_caps_word(x) for x in title.split(' '))

## returns the number of punctuation marks
def get_num_punctuation_marks(title, punctuation_marks={'?', '!'}):
  return sum(title.count(x) for x in punctuation_marks)


## **DECISION BOUNDARY**: the boundary is whether or not the number of capital words that are NOT exceptions is greater than 1. Thats the threshold for whether the program says its "fake" or not
def estimator(x):
  title = x['title']
  return get_num_caps_words(title) >= 1


print(np.mean([estimator(x) == x['is_fake'] for _, x in data.iterrows()]))

data['num_caps_words_title'] = [get_num_caps_words(title) for title in data['title']]
data['num_punctuation_marks_title'] = [get_num_punctuation_marks(title) for title in data['title']]
