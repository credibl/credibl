## functions used for processing data
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup as bs
import requests
from goose3 import Goose
from requests import get

from nltk.corpus import stopwords
import string

# warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# gets the html
def get_html(url):
    response = get(url)
    return response.content
         
# gets the data
# 10/10 documentation would document again
def add_data(url, df):
    response = get(url)
    html = response.content
    extractor = Goose()
    article = extractor.extract(raw_html=response.content)
    text = article.cleaned_text
    title = bs(html, 'html.parser').title
    df['title'] = title
    df['text'] = text


def remove_stopwords(body):
    words = body.split(' ')
    punct = list(string.punctuation)
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.update(punct)
    words = [word for word in words if word not in stopwords_set]
    return ' '.join(words)


def strip_html(body):
    body = bs(body, 'html.parser').text
    body = body.lower()
    return body


def remove_urls(body):
    for word in body:
        if 'www' in word or 'http' in word:
            body = body.replace(word, '')
    return body


def clean_text(body):
    '''
    Cleans the text, removing stopwords, urls, and html tags/stuff
    '''
    if body is None:
        return
    
    body = remove_urls(body)
    body = strip_html(body)
    body = remove_stopwords(body)

    return body
