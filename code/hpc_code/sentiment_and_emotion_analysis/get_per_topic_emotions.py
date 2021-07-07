from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
import json
import time
import ast
import re
import pandas as pd
import sqlite3 as sql
import os
import sys


database_path = '/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
twitter_model_path = '/gpfs0/rami/users/iliapl/data/output_data/lda_models/70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_20_TOPICS'
emotions_path = '/gpfs0/rami/users/iliapl/data/emotion_recognition'
NUM_TOPICS = 20

topics_to_show = [3, 6, 7, 8, 9, 16]

model_dict = Dictionary.load('{}/dict.id2word'.format(twitter_model_path))
corpus = MmCorpus('{}/corpus.mm'.format(twitter_model_path))
model = LdaModel.load('{}/lda.model'.format(twitter_model_path))


T = time.time()

with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'r') as file_handle:
    tweet_topic_map = json.load(file_handle)
    tweet_topic_map = {tweet_id: ast.literal_eval(tup_str) for (tweet_id, tup_str) in
                           tweet_topic_map.items()}

T = time.time() - T
print('Loaded id -> topic dictionary in {} seconds, length {}'.format(T, len(tweet_topic_map)), flush=True)

labels = ['Anger',
          'Disgust',
          'Fear',
          'Joy',
          'Sadness',
          'Surprise']
colors = ['red',
          'brown', 
          'orange', 
          'limegreen', 
          'grey', 
          'deepskyblue']
          
all_emotions_df = pd.read_csv('{}/all_tweet_emotions.csv'.format(emotions_path))

# TEMP
tweet_topic_map = {tweet_id: topic for (tweet_id, topic) in list(tweet_topic_map.items())[:1000]}
# TEMP


topic_aggregated_emotions = {topic: {label: 0 for label in labels} for topic in topics_to_show}
topic_tweet_count = {topic: 0 for topic in topics_to_show}

all_tweet_ids = all_emotions_df['post_id']
all_tweet_ids = all_tweet_ids[~all_tweet_ids.isnull()]
all_tweet_ids = [str(x) for x in list(all_tweet_ids)]

for tweet_id, (topic_id, topic_prob) in tweet_topic_map.items():
    if topic_id not in topics_to_show or tweet_id not in all_tweet_ids:
        continue
        
    for emotion in labels:
      topic_aggregated_emotions[topic_id][emotion] += list(all_emotions_df[all_emotions_df['post_id'] == float(tweet_id)][emotion])[0]
    topic_tweet_count[topic_id] += 1
    
print('topic_tweet_count:', topic_tweet_count)    

for topic in topics_to_show:
    for label in labels:
        topic_aggregated_emotions[topic][label] /= topic_tweet_count[topic]
        
import matplotlib.pyplot as plt
import matplotlib as mpl

# multiple

mpl.rcParams['font.size'] = 18.0

nrows = 2
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 20))


for i, topic_id in enumerate(topics_to_show):
    axes[int(i / ncols)][i % ncols].pie(topic_aggregated_emotions[topic_id].values(), colors=colors, autopct='%1.1f%%', textprops={'fontsize': 18}, shadow=False, radius=1.22)
    axes[int(i / ncols)][i % ncols].set_title(f'Topic {topic_id} Emotions (# tweets: {topic_tweet_count[topic_id]})')

plt.legend(labels, loc='upper center', bbox_to_anchor=(-1.25, -0.1), fancybox=True, shadow=True, ncol=len(labels))

plt.savefig('{}/per_topic_emotions_{}.png'.format(emotions_path, '_'.join([str(i) for i in topics_to_show])))