## functions used for processing data
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from goose import Goose
from requests import get


def get_data(url):
    response = get(url)
    html = response.content
    extractor = Goose()
    article = extractor.extract(raw_html=response.content)
    text = article.cleaned_text 
    data = {
    'url': url,
    'html': html,
    'body': text,
    'is_fake': 0
    }
    
