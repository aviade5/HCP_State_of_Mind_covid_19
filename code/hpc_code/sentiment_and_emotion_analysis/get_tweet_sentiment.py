import pandas as pd
import sqlite3 as sql
import os
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

database_path = "/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db"
output_path = "/gpfs0/rami/users/iliapl/data/output_data/twitter_sentiment_analysis/individual_hcp_53k_authors_2019_2020"

if not os.path.exists(output_path):
    os.makedirs(output_path)

con = sql.connect(database_path)

all_posts = pd.read_sql('SELECT post_id, content FROM posts', con)

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_compound_value(post):
    score = analyzer.polarity_scores(post)
    return score['compound']

T = time.time()

all_posts['sentiment_value'] = all_posts['content'].apply(lambda x: get_sentiment_compound_value(x))

print('Calculated sentiment for {} tweets in {} seconds'.format(len(all_posts), time.time() - T), flush=True)

all_posts.to_csv('{}/tweet_sentiment_values.csv'.format(output_path), index=False)