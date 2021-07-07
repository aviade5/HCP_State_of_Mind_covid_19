import pandas as pd
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import LdaModel
import time
import json
import sqlite3 as sql
import os
import numpy as np
import datetime

twitter_database_path = r'/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
twitter_model_path = r'/gpfs0/rami/users/iliapl/data/output_data/70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_20_TOPICS'

global_confirmed_path = r'/gpfs0/rami/users/iliapl/data/databases/COVID-19/time_series_covid19_confirmed_global.csv'
global_death_path = r'/gpfs0/rami/users/iliapl/data/databases/COVID-19/time_series_covid19_deaths_global.csv'
us_confirmed_path = r'/gpfs0/rami/users/iliapl/data/databases/COVID-19/time_series_covid19_confirmed_US.csv'
us_death_path = r'/gpfs0/rami/users/iliapl/data/databases/COVID-19/time_series_covid19_deaths_US.csv'

output_path = r'/gpfs0/rami/users/iliapl/data/output_data/twitter_over_time/53k_individual_hcps'

start_date = (1, 1, 2020)
end_date = (6, 12, 2020)

country = 'United States'
if country:
    post_ids_path = '/gpfs0/rami/users/iliapl/data/output_data/twitter_location_analysis/author_location_data_53k_authors/post_ids_{}.csv'.format(country.replace(' ', '_'))
    post_ids_df = pd.read_csv(post_ids_path)
    post_ids_df['post_id'] = post_ids_df['post_id'].astype('str')
else:
    post_ids_df = None

NUM_TOPICS = 20


# Load corpus, dictionary and model

T = time.time()

twitter_corpus = MmCorpus('{}/corpus.mm'.format(twitter_model_path))
twitter_dict = Dictionary.load('{}/dict.id2word'.format(twitter_model_path))
twitter_model = LdaModel.load('{}/lda.model'.format(twitter_model_path))

T = time.time() - T
print('Loaded twitter model, corpus, dictionary in {} seconds'.format(T), flush=True)


# Load tweet id to bag-of-words map

T = time.time()
with open('{}/post_id_bow_dict.json'.format(twitter_model_path), 'r') as json_file:
    tweet_id_dict = json.load(json_file)
    print('Tweet ID BOW dictionary size is {}'.format(len(tweet_id_dict)), flush=True)
    if country:
        all_tweet_ids = list(tweet_id_dict.keys())
        wanted_tweets = set(post_ids_df['post_id'])
        for tweet_id in all_tweet_ids:
            if tweet_id not in wanted_tweets:
                tweet_id_dict.pop(tweet_id, None)

        print('After country filtering tweet ID BOW dictionary size is {}'.format(len(tweet_id_dict)), flush=True)


T = time.time() - T
print('Loaded id -> bow dictionary in {} seconds'.format(T), flush=True)

# Load or create tweet_id -> date map

if not os.path.exists(output_path):
    os.makedirs(output_path)

T = time.time()

if not os.path.isfile('{}/tweet_date_map.json'.format(twitter_model_path)) and not country:
    conn = sql.connect(twitter_database_path)
    cur = conn.cursor()
    query = 'SELECT post_id, date FROM posts'
    tweet_id_dates = cur.execute(query).fetchall()
    tweet_id_date_dict = {tweet_id : date for (tweet_id, date) in tweet_id_dates if tweet_id in tweet_id_dict}
    with open('{}/tweet_date_map.json'.format(twitter_model_path), 'w') as file_handle:
        json.dump(tweet_id_date_dict, file_handle)
else:
    with open('{}/tweet_date_map.json'.format(twitter_model_path), 'r') as file_handle:
        tweet_id_date_dict = json.load(file_handle)


T = time.time() - T
print('Loaded id -> date dictionary in {} seconds'.format(T), flush=True)

import ast

# Load or create tweet_id -> topic map

T = time.time()

tweet_id_topic_dict = {}

if not os.path.isfile('{}/tweet_topic_map.json'.format(twitter_model_path)) and not country:
    for tweet_id, bow in tweet_id_dict.items():
        topics = twitter_model.get_document_topics(bow)
        tweet_id_topic_dict[tweet_id] = max(topics, key=lambda tup: tup[1])
        
        # so we can save it as json
        tweet_id_topic_dict[tweet_id] = (tweet_id_topic_dict[tweet_id][0], round(tweet_id_topic_dict[tweet_id][1], 3))
    with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'w') as file_handle:
        json.dump({tweet_id : str(topic) for (tweet_id, topic) in tweet_id_topic_dict.items()}, file_handle)
else:
    with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'r') as file_handle:
        tweet_id_topic_dict = json.load(file_handle)
        tweet_id_topic_dict = {tweet_id : ast.literal_eval(tup_str) for (tweet_id, tup_str) in tweet_id_topic_dict.items()}
    
if country:
    all_tweet_ids = list(tweet_id_topic_dict.keys())
    wanted_tweets = set(post_ids_df['post_id'])
    for tweet_id in all_tweet_ids:
        if tweet_id not in wanted_tweets:
            tweet_id_topic_dict.pop(tweet_id, None)
        
    print('Tweet ID topic dictionary size is {}'.format(len(tweet_id_topic_dict)), flush=True)
    print('After country filtering tweet ID topic dictionary size is {}'.format(len(tweet_id_topic_dict)), flush=True)
    
T = time.time() - T
print('Loaded id -> topic dictionary in {} seconds'.format(T), flush=True)

import datetime

# covid data starts on 22/01
covid_data_start_date_obj = max(datetime.date(start_date[2], start_date[1], start_date[0]), datetime.date(2020, 1, 22))

covid_data_end_date_obj = datetime.date(end_date[2], end_date[1], end_date[0])

global_confirmed_df = pd.read_csv(global_confirmed_path)
global_death_df = pd.read_csv(global_death_path)
us_confirmed_df = pd.read_csv(us_confirmed_path)
us_death_df = pd.read_csv(us_death_path)


def get_new_cases_per_day(case_type, country_name, state, covid_data_start_date, end_date):
    if case_type != 'confirmed' and case_type != 'death':
        print('Valid cases are confirmed and death')
        
    if case_type == 'confirmed':
        if state:
            df = us_confirmed_df
        else:
            df = global_confirmed_df
    elif case_type == 'death':
        if state:
            df = us_death_df
        else:
            df = global_death_df
        
    series = []
    
    start_date = covid_data_start_date
    delta = datetime.timedelta(days=1)

    date = start_date
    
    while date <= end_date:
        
        date_str = date.strftime('%m/%d/%y').lstrip("0").replace("/0", "/")    
        previous_day_str = (date - delta).strftime('%m/%d/%y').lstrip('0').replace('/0', '/')
        
        if country_name == 'global':
            series.append(sum(df[date_str]) - sum(df[previous_day_str]))
        else:
            column_name = 'Country/Region' if not state else 'Province_State'
            row = df.loc[df[column_name] == country_name]
            
            # sum is applied because some countries (like UK) have provinces, creating duplicate country names
            daily_count = int(sum(row[date_str]) - sum(row[previous_day_str]))
            series.append(daily_count)
                
        date += delta
    return series

def get_tweets_per_day(country_name, state, topic_id, start_date, end_date):
    if topic_id != -1:
        tweet_ids = [tweet_id for (tweet_id, tweet_topic) in tweet_id_topic_dict.items() if tweet_topic[0] == topic_id]
    else:
        tweet_ids = [tweet_id for (tweet_id, _) in tweet_id_topic_dict.items()]
    tweets_per_day = []
    delta = end_date - start_date
    num_days = delta.days
    
    for i in range(num_days + 1):
        tweets_per_day.append([])
    
    print('Getting tweets per day for {} (state: {}), topic id: {}, start_date: {}, end_date: {}'.format(country_name,
                                                                                                        state, topic_id,
                                                                                                        start_date, end_date), flush=True)
    for tweet_id in tweet_ids:
        if tweet_id not in tweet_id_date_dict:
            continue
        tweet_date = tweet_id_date_dict[tweet_id]
        tweet_date = tweet_date[:tweet_date.index(' ')] # keep just the date, without time, ,%Y-%m-%d
        date_obj = datetime.datetime.strptime(tweet_date, '%Y-%m-%d').date()
        if start_date <= date_obj <= end_date:
            day_idx = (date_obj - start_date).days
            tweets_per_day[day_idx].append(tweet_id)
    
    
    if country_name != 'global':
        country_state_dict = tweet_id_country_dict
        if state:
            country_name = state_name_code_dict.get(country_name, None)
            if country_name is None or country_name not in set(tweet_id_state_dict.values()):
                raise ValueError('Invalid state name')
                return None
            
            # country_name is the name of a state
            country_state_dict = tweet_id_state_dict
            
        country_tweets_per_day = []
        for day_tweets in tweets_per_day:
            new_day_tweets = []
            for tweet in day_tweets:
                if tweet in country_state_dict and country_state_dict[tweet] == country_name:
                    new_day_tweets.append(tweet)
            country_tweets_per_day.append(new_day_tweets)
        tweets_per_day = country_tweets_per_day
    
    return tweets_per_day
    
def get_num_different_authors(tweet_list):
    return len(set([tweet_id_author_dict[tweet_id] for tweet_id in tweet_list]))
    
def get_num_tweets_per_day(country_name, state, topic_id, start_date, end_date, num_distinct_authors=False):
    tweets_per_day = get_tweets_per_day(country_name, state, topic_id, start_date, end_date)
    num_different_authors = None
    if get_num_different_authors:
        num_different_authors = [get_num_different_authors(daily_tweets) for daily_tweets in tweets_per_day]
    return [len(lst) for lst in tweets_per_day], num_different_authors

def get_topic_tweets_per_week(model, corpus, topic, start_day, start_month, start_year, end_day, end_month, end_year):
    delta = datetime.timedelta(days=6)
    
    all_week_tweets = []
    
    start_date = datetime.date(start_year, start_month, start_day)
    end_date = datetime.date(end_year, end_month, end_day)
    current_date = start_date
    
    while current_date <= end_date:
        
        end_of_week = min(current_date + delta, end_date)
        
        # flatten list of daily tweets
        all_week_tweets.append([tweet for day_tweets in get_tweets_per_day('global', False, topic,
                                current_date, end_of_week)
                                for tweet in day_tweets])
        
        current_date += (delta + datetime.timedelta(days=1))
    return all_week_tweets
    
print('Starting to collect tweets per topic per week', flush=True)    

tweets_per_topic_per_week = {}

T = time.time()

for topic in list(range(NUM_TOPICS)):
    tweets_per_topic_per_week[topic] = get_topic_tweets_per_week(twitter_model, twitter_corpus, topic, *start_date, *end_date)
    print()
    print('Finished {}/{} topics in {} seconds'.format(topic + 1, NUM_TOPICS, time.time() - T), flush=True)
    print()
    
print('Creating tweet count dictionary...')
    
T = time.time()
    
tweet_count_per_topic_per_week = {topic_id: [len(weekly_tweets) for weekly_tweets in topic_week_tweets] for topic_id, topic_week_tweets in tweets_per_topic_per_week.items()}

print('Finished in {} seconds'.format(time.time() - T), flush=True)

num_weeks = len(list(tweet_count_per_topic_per_week.items())[0][1])
start_date_obj = datetime.date(*(reversed(start_date)))
d = {'t{}'.format(i) : tweet_count_per_topic_per_week[i] for i in range(NUM_TOPICS)}
df = pd.DataFrame(d, index=[start_date_obj + datetime.timedelta(days=i*7) for i in range(num_weeks)])
df.to_csv('{}/tweet_topic_count_per_week{}.csv'.format(output_path, '' if not country else '_{}'.format(country.replace(' ', '_'))))

print('Saved tweet_topic_count_per_week{}.csv to {}'.format(output_path, '' if not country else '_{}'.format(country.replace(' ', '_'))), flush=True)