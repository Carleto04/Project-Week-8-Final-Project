#!/usr/bin/env python
# coding: utf-8

# In[73]:


#import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pickle

def text_preproc(text):
    # auxiliar function to delete stopwords
    def stopword_del(sentence):
        stopwords_list = stopwords.words("english")
        word_tokens = word_tokenize(sentence)
        tokens_without_sw = [word for word in word_tokens if not word in stopwords_list]
        str1 = ' '.join(tokens_without_sw)
        return str1

    # auxiliar function to remove a pattern defined by a regular expression 
    def remove_by_regex(tweet, regexp):
            return re.sub(regexp, '', tweet)

    # 3 specific cleaning functions to remove numbers, url's and special characters
    def remove_numbers(tweet):
        return remove_by_regex(tweet, re.compile(r"[1234567890]"))

    def remove_url(tweet):
        return remove_by_regex(tweet, re.compile(r"http.?://[^\s]+[\s]?"))

    def remove_special_char(tweet):
        return re.sub(r"[^a-zA-Z0-9 ]", "", tweet) #add space placeholder

    # general cleaning function to do it all at once
    def clean_up(tweet):
        tweet = remove_numbers(tweet)
        tweet = remove_url(tweet)
        tweet = remove_special_char(tweet)
        return tweet.lower().strip()

    
    #apply previously defined functions all at once
    text = stopword_del(text)
    text = clean_up(text)
    
    df = pd.DataFrame(data={'content': text}, index=[0])
    
    #stem the words in the sentences and delete too-large-whitespaces
    stemmer = SnowballStemmer("english")
    df["content_stemmed"] = stemmer.stem(text)
    df["content_stemmed"] = [' '.join(text.split())]
    
    #loadtransformer
    text_vectorizer = pickle.load(open("text_vectorizer_model.sav", "rb"))
    
    #convert the words in sentences to a new dataframe of vectors
    vector_df = text_vectorizer.transform([df["content_stemmed"][0]])
    
    return vector_df

