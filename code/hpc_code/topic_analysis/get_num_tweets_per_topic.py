from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
import json
import time
import ast
import re
import pandas as pd
import sqlite3 as sql
import os
import datetime
import pytz


database_path = '/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
twitter_model_path = '/gpfs0/rami/users/iliapl/data/output_data/lda_models/70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_20_TOPICS'
location_analysis_path = '/gpfs0/rami/users/iliapl/data/output_data/twitter_location_analysis/POI_Followers_13-06-20'
NUM_TOPICS = 20

model_dict = Dictionary.load('{}/dict.id2word'.format(twitter_model_path))
corpus = MmCorpus('{}/corpus.mm'.format(twitter_model_path))
model = LdaModel.load('{}/lda.model'.format(twitter_model_path))


T = time.time()

with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'r') as file_handle:
    tweet_id_topic_dict = json.load(file_handle)
    tweet_id_topic_dict = {tweet_id: ast.literal_eval(tup_str) for (tweet_id, tup_str) in
                           tweet_id_topic_dict.items()}

T = time.time() - T
print('Loaded id -> topic dictionary in {} seconds'.format(T), flush=True)

topic_tweet_count = {topic_id: 0 for topic_id in range(NUM_TOPICS)}

for i, (tweet_id, topic_prob) in enumerate(tweet_id_topic_dict.items()):
  if i % 100000 == 0:
    print('Finished {}/{} tweets'.format(i, len(tweet_id_topic_dict)), flush=True)
  topic_tweet_count[topic_prob[0]] += 1

pd.Series([topic_tweet_count[i] for i in range(NUM_TOPICS)]).to_csv('{}/topic_tweet_count.csv'.format(twitter_model_path))

print('Saved csv to {}'.format(twitter_model_path), flush=True)