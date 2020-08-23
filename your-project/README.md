<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Mozart's Sister
*by Cristina Arias and Carles Rosell*

*[Data Analysis, Ironhack Barcelona & 19-08-2020]*

## Content
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Cleaning](#cleaning)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Links](#links)

## Project Description
Mozart's Sister can recommend you songs in Spotify based on the emotions she detects you are experiencing from having a short written conversation with you.

## Dataset
To teach Mozart's Sister how to recognise emotions we have used a dataset with sentences linked to different emotions obtained from the work presented on the 
paper "Contextualized Affect Representations for Emotion Recognition" by Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, Yi-Shin Chen 
(https://www.aclweb.org/anthology/D18-1404/).

This dataset is constituted by:
- 18K sentences that we have splitted in 16K for model training and 2K for model testing.
- These sentences are linked to the emotion tags 'love', 'joy', 'surprise', 'sadness', 'fear' and 'anger'.


The folder structure is as follows:
1. Data: Contains the training and test set as well as the spotify data (.txt and .csv).
2. YourProject: Final_prediction_model (model and accuracy, .ipynb); Text_cleaner (cleaning and vectorization function, .ipynb, .py), 
	Skeleton_appVF (interface, .ipynb, .py), track_recommender (spotify connection function, .ipynb, .py); 
	Emotion_detect_model and Text_vectorizer_model (trained model, .sav);
3. WIP: others

## Cleaning
To feed the data into the algorithm, we first remove the noise from the sentences. We achieve this by using the regular expressions module to remove unwanted 
characters like spaces and symbols, as well as stopwords and stemming.

We then isolate the words relevant to the emotions through tokenization and translate them into a language that Mozart's Sister can understand through vectorization. 
What happens in this step is the following:
a) Each word is turned into a feature
b) Each sentence is assigned to the different features


## Model Training and Evaluation
To train Mozart's Sister we have used a Natural Language Process algorithm called Support Vector Machine (applying multilabel Crammer-Singer, which is a one-vs-rest 
classification method). This algorithm allows to apply supervised learning methodology.

The objective of the Support Vector Machine algorithm is to find a hyperplane in an n-dimensional space (n being the number of features) that distinctly classifies 
the data points and predicts the appropriate label.

Test_set prediction capability of Mozart's Sister currently at 89%.


## Future Work
1. Even when it is unlikely due to the wide spotify selection available, it would be useful to incorporate user's feedback into the algorithm, so that when the user 
dislikes a recommendation, it is not displayed again.
2. Creating a chatbot conversation based on NLP.
3. Improving the emotion prediction capability of Mozart's Sister. It would also be useful to improve her detection of full usual expressions.

## Workflow
A. User name collection
B. Greetings
C. Emotion detection
D. Recommendation
E. Feedback
F. Loop


## Links
[Repository](https://github.com/Carleto04/Project-Week-8-Final-Project)  
[Slides](https://slides.com/cris-arias/deck-8ae022)  
[Trello](https://trello.com/b/x8FKuzHh/projectfinal-your-face-is-a-poem)  
