#!/usr/bin/env python
# coding: utf-8

# In[3]:


def recommender(user_sent):

    #libraries to use spotipy
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    import getpass
    import requests
    import json
    import pandas.io.json as json_normalize
    import pandas as pd
    import numpy as np
    import random

    #import dataset from spotify API
    spoty_mood = pd.read_csv("../Data/spotify_data.csv")
    
    #drop unnecessary columns from dataset
    spoty_mood_clean = spoty_mood.drop(['acousticness', 'danceability', 'duration_ms', 'energy',
           'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
           'mode', 'name', 'popularity', 'release_date', 'speechiness', 'tempo', 'year'], axis=1)
    
    #cathegorize songs with valence to sentiments
    bins = [-np.inf, 0.33, 0.66, np.inf]
    labels = ['negative', 'neutral', 'positive']
    spoty_mood_clean['sentiment'] = pd.cut(spoty_mood_clean['valence'], labels=labels, bins=bins)


    # To keep token secured
    SPOTIPY_CLIENT_ID = '97f62274085f41b69f9fe2d69b8aea43'
    SPOTIPY_CLIENT_SECRET = getpass.getpass(prompt='Enter-your-token-here')

    #activate spotipy
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id = SPOTIPY_CLIENT_ID, 
                                                                                  client_secret = SPOTIPY_CLIENT_SECRET))

    #tracks lists grouped by sentiment
    positive_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "positive"]
    negative_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "negative"]
    neutral_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "neutral"]

    #choose randomly a track_ID from sentiments track lists
    track_pos = random.choice(positive_tracks_lst)
    track_neg = random.choice(negative_tracks_lst)
    track_neu = random.choice(neutral_tracks_lst)
    
    #find the track through track_ID
    recomm_answ_pos = spotify.track(track_id=track_pos)
    recomm_answ_neg = spotify.track(track_id=track_neg)
    recomm_answ_neu = spotify.track(track_id=track_neu)
    
    #output the name, singer and link to recommendation depending on the sentiment
    if user_sent == "positive":
        print('track    : ', recomm_answ_pos['name'])
        print('artist    :', recomm_answ_pos['album']['artists'][0]['name'])
        print('audio    : ', recomm_answ_pos['external_urls']['spotify'])
    elif user_sent == "negative":
        print('track    : ', recomm_answ_neg['name'])
        print('artist    :', recomm_answ_neg['album']['artists'][0]['name'])
        print('audio    : ', recomm_answ_neg['external_urls']['spotify'])
    else:
        print('track    : ', recomm_answ_neu['name'])
        print('artist    :', recomm_answ_neu['album']['artists'][0]['name'])
        print('audio    : ', recomm_answ_neu['external_urls']['spotify'])

