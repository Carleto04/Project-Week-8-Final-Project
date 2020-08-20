#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import random
from text_cleaner import text_preproc
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from track_recommender import recommender
import pickle
model = pickle.load(open("emotion_detect_model.sav", 'rb'))


# # Lists

# In[2]:


# greetings list
starting_q = ["Why do you answer that way", "How are you today", "How is your week going", "What have you done today", "What word would you choose for today", "If I say hey, you say...", "Do you want to play a game", "What is your plan today"]
question = random.choice(starting_q)

# sentiments list
positive_lst = ['yes', 'ok', 'fine', 'good', 'excellent', 'awesome', 'amazing', 'great', 'positive', 'agree', 'agreed', 'absolutely','accepted','acclaimed','accomplish','accomplishment', 'achievement', 'active', 'admire', 'adorable', 'adventure', 'affirmative', 'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'attractive', 'awesome', 'beaming', 'beautiful','believe','beneficial','bliss','bountiful', 'bounty', 'brave','bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain','champ','champion','charming','cheery','choice','classic','classical', 'clean', 'commend', 'composed', 'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 'efficient', 'effortless','electrifying','elegant','enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'excellent', 'exciting', 'exquisite', 'fabulous', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'laugh', 'legendary', 'light', 'lively', 'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'nurturing', 'nutritious', 'okay', 'one-hundred percent', '100%', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasant', 'pleasurable', 'plentiful', 'poised', 'polished', 'popular', 'positive', 'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stunning', 'stupendous', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 'thorough', 'thrilling', 'thriving', 'tops', 'tranquil', 'transformative', 'transforming', 'trusting', 'truthful', 'unreal', 'unwavering', 'up', 'upbeat', 'upright', 'upstanding', 'valued', 'vibrant', 'victorious', 'victory', 'vigorous','virtuous', 'vital', 'vivacious', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'wondrous', 'worthy', 'wow', 'yes', 'yummy', 'zeal', 'zealous']
negative_lst = ['no', 'bad', 'worse', 'worst', 'evil', 'devil', 'negative', 'sucks', 'loss', 'abominable', 'aching', 'afflicted', 'affraid', 'aggressive', 'agonized', 'alarmed', 'alienated', 'alone', 'angry', 'anguish', 'annoyed', 'anxious', 'appalled', 'bad', 'bitter', 'boiling', 'bored', 'betrayed', 'ashamed', 'abnormal', 'cold', 'cowardly', 'cross', 'crushed', 'complaining', 'cheated', 'confused', 'crappy', 'dejected', 'depressed', 'deprived', 'desolate', 'despair', 'desperate', 'despicable', 'detestable', 'diminished', 'disappointed', 'discouraged', 'disgusting', 'disillusioned', 'disinterested', 'dismayed', 'dissatisfied', 'distressed', 'distrustful', 'dominated', 'doubtful', 'dull', 'embarrassed', 'empty', 'enraged', 'evil', 'excluded', 'exiled', 'fatigued', 'fearful', 'forced', 'frightened', 'frustrated', 'fuming', 'grief', 'grieved', 'guilty', 'hateful', 'heartbroken', 'helpless', 'hesitant', 'hostile', 'humiliated', 'hurt', 'despair', 'incapable', 'incensed', 'indecisive', 'indifferent', 'indignant', 'inferior', 'inflamed', 'infuriated', 'injured', 'insensitive', 'indelicate', 'insulting', 'irritated', 'lifeless', 'lonely', 'lost', 'lousy', 'liar', 'lame', 'livid', 'menaced', 'miserable', 'misgiving', 'mournful', 'misunderstood', 'manipulated', 'nervous', 'neutral', 'nonchalant', 'negated', 'offended', 'offensive', 'objected', 'overwhelmed', 'obstructed', 'pained', 'panic', 'paralyzed', 'pathetic', 'perplexed', 'pessimistic', 'powerless', 'preoccupied', 'provoked', 'quaking', 'questioned', 'rejected', 'repugnant', 'resentful', 'reserved', 'restless', 'sad', 'scared', 'shaky', 'shy', 'skeptical', 'sore', 'sorrowful', 'stupefied', 'sulky', 'suspicious', 'tearful', 'tense', 'terrible', 'terrified', 'threatened', 'timid', 'tormented', 'tortured', 'tragic', 'unbelieving', 'uncertain', 'uneasy', 'unhappy', 'unpleasant', 'unsure', 'upset', 'useless', 'unloved', 'unimportant', 'unconnected', 'victimized', 'worthless', 'worthiness', 'wary', 'weary', 'woeful', 'worried', 'wronged']

# user sentiment today
user_sent = []
user_sent2 = []
sentiment_lst = ['anger', 'sadness', 'fear', 'surprise', 'joy', 'love']
vals_to_replace = {0:'anger', 1:'sadness', 2:'fear', 3:'surprise', 4:'joy', 5:'love'}


# # Greetings

# In[3]:


# define username
user_name = input("Hello, what is your name? ").capitalize()
print(f'Hello {user_name}')


# In[4]:


# ask how is the user and append emotion

while len(user_sent) == 0:
    q1 = input(f"{question} {user_name}? ").lower()
    text = text_preproc(q1)
    sentiment_key = model.predict(text)
    sentiment = vals_to_replace.get(model.predict(text)[0])
    user_sent.append(sentiment)


# In[5]:


# sentiment confirmation
sent_conf = input(f"So you are feeling {user_sent[0]}? ").lower()
while sent_conf not in positive_lst and sent_conf not in negative_lst:
    sent_conf = input(f"Are you feeling {user_sent[0]}? ").lower()
        
# store sentiment if ML fails
if sent_conf in negative_lst:
    sent_q = input("Then today you are feeling anger, sadness, fear, surprise, joy or love? ").lower()
    while sent_q not in sentiment_lst:
        sent_q = input("Then how are you feeling today? (answer anger, sadness, fear, surprise, joy or love) ").lower()
    user_sent[0] = sent_q
    
# reinforce or change sentiment
sent_ch = input(f"Do you want to keep or change your {user_sent[0]} sentiment? ").lower()
while sent_ch != "change" and sent_ch != "keep":
    sent_ch = input(f"Do you want to keep or change your {user_sent[0]} sentiment? (type keep or change) ").lower()


# In[6]:


# feed chatbot and get answer according to sentiment
# sentiment_lst = ['anger', 'sadness', 'fear', 'surprise', 'joy', 'love']

if user_sent[0] == "anger" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'It is scary that you are feeling {user_sent2[0]}. To release some adrenaline listen to this:')
    print(recommender(user_sent2[0]))
    
elif user_sent[0] == "anger" and sent_ch == "change":
    user_sent2.append("love")
    print(f'What about some {user_sent2[0]} instead? Listen to this for transformation:')
    print(recommender(user_sent2[0]))
    
elif user_sent[0] == "love" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'All you need is love!!! Listen to this:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "love" and sent_ch == "change":
    user_sent2.append("anger")
    print(f'You are following a dark path, young Padawan. For joining the evil Empire, listen to this:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "sadness" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'For wallowing into your misery listen to this:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "sadness" and sent_ch == "change":
    user_sent2.append("joy")
    print(f'I can perform miracles. Listen to this and get convinced:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "joy" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'That is what makes life worth it. I know by experience. Listen to this and dance, maybe:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "joy" and sent_ch == "change":
    user_sent2.append("sadness")
    print(f'It is ok to cry. I will cry with you through this song:')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "fear" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'Fear leads to anger. Anger leads to hate. Hate leads to suffering. Keep listening to my tunes to regulate:')
    print (recommender(user_sent2[0]))

elif user_sent[0] == "fear" and sent_ch == "change":
    user_sent2.append("surprise")
    print(f'So you want to go to the top floor? Did you learn nothing from the movies??')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "surprise" and sent_ch == "keep":
    user_sent2.append(user_sent[0])
    print(f'Mystery tune...')
    print(recommender(user_sent2[0]))

elif user_sent[0] == "surprise" and sent_ch == "change":
    user_sent2.append("fear")
    print(f'Oh oh... Surprise just got sour. Now feel the FEAR:')
    print(recommender(user_sent2[0]))


# In[7]:


# feedback from recommendations
feedback1 = input('Did you like it? ').lower()
while feedback1 not in positive_lst and feedback1 not in negative_lst:
    feedback1 = input('Come on! Did you like it? ').lower()
    
# feedback if user want to keep on going with recommendations
#if not, end of the program
if feedback1 in positive_lst:
    feedback2 = input("Great! Do you want more tips?").lower()
    while feedback2 not in positive_lst and feedback2 not in negative_lst:
        feedback2 = input("Do you want more tips? ").lower()
elif feedback1 in negative_lst:
    feedback2 = input("Sorry to hear that. May I recommend something different? ").lower()
    while feedback2 not in positive_lst and feedback2 not in negative_lst:
        feedback2 = input("May I recommend something different? ").lower()


# In[8]:


# loop for recommending more tips
while feedback2 in positive_lst:
    print(f"So you want more ;-) Let's keep the vibe on!")
    print(recommender(user_sent2[0]))

    
    feedback1 = input('Did you like it? ').lower()
    while feedback1 not in positive_lst and feedback1 not in negative_lst:
        feedback1 = input('Come on! Did you like it? ').lower()
    
    if feedback1 in positive_lst:
        feedback2 = input("Great! Do you want more tips?").lower()
        while feedback2 not in positive_lst and feedback2 not in negative_lst:
            feedback2 = input("Do you want more tips? ").lower()
    elif feedback1 in negative_lst:
        feedback2 = input("You killed my vibe :-( Eitherway, may I recommend something different? ").lower()
        while feedback2 not in positive_lst and feedback2 not in negative_lst:
            feedback2 = input("Do you want more tips? ").lower()
            
print(f"Ok {user_name}. See you later")


# In[ ]:




