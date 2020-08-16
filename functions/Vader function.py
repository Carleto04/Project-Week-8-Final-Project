#!/usr/bin/env python
# coding: utf-8

# ## Out of the box Sentiment Score

# In[52]:


def vader(x):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(x)

