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

NUM_TWEETS_PER_TOPIC = int(sys.argv[1])

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

con = sql.connect(database_path)
tweet_id_content_df = pd.read_sql('SELECT post_id, content FROM posts', con)

# PER TOPIC: Create dataframe containing tweet_id and fit value
# Sort dataframe by fit value
# Add column of content according to (tweet_id, content) dataframe that was loaded

final_per_topic_dfs = [pd.DataFrame(columns=['tweet_id', 'fit_value']) for i in range(NUM_TOPICS)]

T = time.time()

for i, (tweet_id, (topic_id, topic_prob)) in enumerate(tweet_id_topic_dict.items()):
  if i % 100000 == 0:
    print('Finished {}/{} tweets in {} seconds'.format(i, len(tweet_id_topic_dict), time.time() - T), flush=True)
  if not (0 <= topic_id < NUM_TOPICS):
    continue
  final_per_topic_dfs[topic_id] = final_per_topic_dfs[topic_id].append({'tweet_id': tweet_id, 'fit_value': topic_prob}, ignore_index=True)

final_per_topic_dfs = [df.sort_values(by=['fit_value'], axis=0, ascending=False, ignore_index=True).reset_index(drop=True) for df in final_per_topic_dfs]

final_per_topic_dfs = [df.head(NUM_TWEETS_PER_TOPIC) for df in final_per_topic_dfs]

# not enough tweets for some topic
if len(set([len(df) for df in final_per_topic_dfs])) > 1:
  min_df_length = min([len(df) for df in final_per_topic_dfs])
  print('Trimming tweet count per topic to', min_df_length)
  final_per_topic_dfs = [df.head(min_df_length) for df in final_per_topic_dfs]

# add column of content per each df
for df in final_per_topic_dfs:
  # x is a string cause its taken from the json
  # tweet_id_content_df['post_id'] is a series of floats
  
  df['content'] = df['tweet_id'].apply(lambda x: list(tweet_id_content_df[tweet_id_content_df['post_id'] == float(x)]['content'])[0])
    
final_df = pd.DataFrame({'topic{}'.format(i): list(final_per_topic_dfs[i]['content']) for i in range(NUM_TOPICS)})

final_df.to_csv('{}/top_{}_topic_tweets.csv'.format(twitter_model_path, NUM_TWEETS_PER_TOPIC), index=False)
  