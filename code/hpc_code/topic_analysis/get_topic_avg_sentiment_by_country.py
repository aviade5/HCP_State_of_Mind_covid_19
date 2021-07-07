import pandas as pd
import time
import json
import ast

twitter_model_path = r'/gpfs0/rami/users/iliapl/data/output_data/70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_20_TOPICS'

country = None
if country:
    post_ids_path = '/gpfs0/rami/users/iliapl/data/output_data/twitter_location_analysis/author_location_data_53k_authors/post_ids_{}.csv'.format(country.replace(' ', '_'))
    post_ids_df = pd.read_csv(post_ids_path)
    post_ids_df['post_id'] = post_ids_df['post_id'].astype('str')
else:
    post_ids_df = None

NUM_TOPICS = 20

with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'r') as json_file:
    tweet_topic_map = json.load(json_file)
    tweet_topic_map = {tweet_id : ast.literal_eval(tup_str) for (tweet_id, tup_str) in tweet_topic_map.items()}

tweet_sentiment_df = pd.read_csv('/gpfs0/rami/users/iliapl/data/output_data/twitter_sentiment_analysis/individual_hcp_53k_authors_2019_2020/tweet_sentiment_values.csv')

T = time.time()

tweet_sentiment_df['post_id'] = tweet_sentiment_df['post_id'].astype('str')

tweet_ids = list(tweet_topic_map.keys())

if country:
  country_tweets = set(post_ids_df['post_id'])
  for tweet_id in tweet_ids:
      if tweet_id not in country_tweets:
          tweet_topic_map.pop(tweet_id, None)
          
print('Tweet topic map size is {}'.format(len(tweet_topic_map)), flush=True)

tweet_sentiment_df = tweet_sentiment_df[tweet_sentiment_df['post_id'].isin(pd.Series(tweet_topic_map.keys()))]

tweet_sentiment_df['topic'] = tweet_sentiment_df['post_id'].apply(lambda tweet_id: tweet_topic_map[tweet_id][0])

print('Added topic column to sentiment df containing {} tweets in {} seconds'.format(len(tweet_sentiment_df), time.time() - T), flush=True)

avg_sentiment_values = []

for topic_id in range(NUM_TOPICS):
    # also save all sentiment values per topic for further statistical analysis
    sentiment_values_series = tweet_sentiment_df[tweet_sentiment_df['topic'] == topic_id]['sentiment_value']
    
    avg_sentiment_values.append(sentiment_values_series.mean())
    
    sentiment_values_series.to_csv('{}/topic_{}_sentiment_values.csv'.format(twitter_model_path, topic_id));

print(avg_sentiment_values, flush=True)

if country:
  pd.DataFrame({'topic_avg_sentiment': avg_sentiment_values}).to_csv('{}/topic_avg_sentiment_{}.csv'.format(twitter_model_path, country.replace(' ', '_')))
else:
  pd.DataFrame({'topic_avg_sentiment': avg_sentiment_values}).to_csv('{}/topic_avg_sentiment.csv'.format(twitter_model_path))  