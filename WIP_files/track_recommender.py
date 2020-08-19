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
    bins = [-np.inf, 0.166, 0.33, 0.5, 0.66, 0.83, np.inf]
    labels = ['anger', 'sadness', 'fear', 'surprise', 'joy', 'love']
    spoty_mood_clean['sentiment'] = pd.cut(spoty_mood_clean['valence'], labels=labels, bins=bins)


    # To keep token secured
    SPOTIPY_CLIENT_ID = '97f62274085f41b69f9fe2d69b8aea43'
    SPOTIPY_CLIENT_SECRET = getpass.getpass(prompt='Enter-your-token-here')

    #activate spotipy
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id = SPOTIPY_CLIENT_ID, 
                                                                                  client_secret = SPOTIPY_CLIENT_SECRET))

    #tracks lists grouped by sentiment
    anger_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "anger"]
    sadness_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "sadness"]
    fear_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "fear"]
    surprise_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "surprise"]
    joy_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "joy"]
    love_tracks_lst = [spoty_mood_clean['id'][track] for track in range(len(spoty_mood_clean)) 
                           if spoty_mood_clean['sentiment'][track] == "love"]
    
    
    #choose randomly a track_ID from sentiments track lists
    track_anger = random.choice(anger_tracks_lst)
    track_sadness = random.choice(sadness_tracks_lst)
    track_fear = random.choice(fear_tracks_lst)
    track_surprise = random.choice(surprise_tracks_lst)
    track_joy = random.choice(joy_tracks_lst)
    track_love = random.choice(love_tracks_lst)
    
    #find the track through track_ID
    recommendations = {'anger': spotify.track(track_id=track_anger), 'sadness': spotify.track(track_id=track_sadness), 'fear': spotify.track(track_id=track_fear), 'surprise': spotify.track(track_id=track_surprise), 'joy': spotify.track(track_id=track_joy), 'love': spotify.track(track_id=track_love)}
    
    #output the name, singer and link to recommendation depending on the sentiment
    print('track    : ', recommendations.get(user_sent)['name'])
    print('artist    :', recommendations.get(user_sent)['album']['artists'][0]['name'])
    print('audio    : ', recommendations.get(user_sent)['external_urls']['spotify'])