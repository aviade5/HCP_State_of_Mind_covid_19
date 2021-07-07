import pandas as pd
import numpy as np
import sqlite3 as sql
import os
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

database_path = "/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db"
output_path = "./tweets_to_classify"

if not os.path.exists(output_path):
    os.makedirs(output_path)

con = sql.connect(database_path)

all_posts = pd.read_sql('SELECT post_id, content FROM posts', con)

# only 5 cause two nodes have some issue with tensorflow/theano or are very slow
divided_dfs = np.array_split(all_posts, 5)

for i, df in enumerate(divided_dfs):
  df.to_csv('{}/job{}_tweets.csv'.format(output_path, i + 1), index=False)