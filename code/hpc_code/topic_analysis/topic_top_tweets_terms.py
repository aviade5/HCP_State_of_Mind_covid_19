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


with open('{}/post_id_bow_dict.json'.format(twitter_model_path), 'r') as file_handle:
    tweet_id_bow_dict = json.load(file_handle)

T = time.time()

tweet_id_topic_dict = {}

if not os.path.isfile('{}/tweet_topic_map.json'.format(twitter_model_path)):
    for tweet_id, bow in tweet_id_bow_dict.items():
        topics = model.get_document_topics(bow)
        tweet_id_topic_dict[tweet_id] = max(topics, key=lambda tup: tup[1])

        # so we can save it as json
        tweet_id_topic_dict[tweet_id] = (tweet_id_topic_dict[tweet_id][0], round(tweet_id_topic_dict[tweet_id][1], 3))
    with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'w') as file_handle:
        json.dump({tweet_id: str(topic) for (tweet_id, topic) in tweet_id_topic_dict.items()}, file_handle)
else:
    with open('{}/tweet_topic_map.json'.format(twitter_model_path), 'r') as file_handle:
        tweet_id_topic_dict = json.load(file_handle)
        tweet_id_topic_dict = {tweet_id: ast.literal_eval(tup_str) for (tweet_id, tup_str) in
                               tweet_id_topic_dict.items()}

T = time.time() - T
print('Loaded id -> topic dictionary in {} seconds'.format(T), flush=True)

import time
import os
import ast

T = time.time()

tweet_id_country_dict = {}

with open('{}/tweet_countries.json'.format(location_analysis_path), 'r') as file_handle:
    tweet_id_country_dict = json.load(file_handle)

tweet_id_state_dict = {}

with open('{}/tweet_states.json'.format(location_analysis_path), 'r') as file_handle:
    tweet_id_state_dict = json.load(file_handle)

with open('{}/state_code_name.json'.format(location_analysis_path), 'r') as file_handle:
    state_code_name_dict = json.load(file_handle)

T = time.time() - T
print('Loaded id -> country/state dictionaries in {} seconds'.format(T), flush=True)

conn = sql.connect(database_path)

def get_topn_tweets_for_topic(topic_id, topn):
    topic_tweets = [(tweet_id, prob) for (tweet_id, (tweet_topic_id, prob)) in tweet_id_topic_dict.items() if
                    tweet_topic_id == topic_id]
    sorted_tweets = sorted(topic_tweets, key=lambda tup: tup[1], reverse=True)

    # we retrieve this number of tweets in hope that there will actually be topn distinct ones out of them
    max_tweets_to_retrieve = 200

    tweet_contents = []
    query = "SELECT post_id, author, date, url, content FROM posts WHERE post_id IN {}".format(
        str(tuple([tweet_info[0] for tweet_info in sorted_tweets[:max_tweets_to_retrieve]])))

    cur = conn.cursor()
    results = cur.execute(query).fetchall()

    results = [(result[0], tweet_id_topic_dict[str(result[0])][1], result[1], result[2], result[3],
                re.sub('RT @.*: ', '', result[4]).replace('\n', ' ').replace('\r\n', ' ').replace('\t', ' ')) for result
               in results]
    sorted_results = sorted(results, key=lambda tup: tup[1], reverse=True)
    unique_contents = set([result[5] for result in sorted_results])

    topn_results = []
    for result in sorted_results:
        if result[5] in unique_contents:
            topn_results.append(result)
            unique_contents.remove(result[5])
            if len(topn_results) == topn:
                break

    if len(topn_results) < topn:
        print(
            'Could not get requested topn for topic {}! Only got {}/{}! Increase max_tweets_to_retrieve value!'.format(
                topic_id, len(topn_results), topn), flush=True)

    for result in topn_results:
        post_id = result[0]
        if post_id not in tweet_id_country_dict:
            location = 'unknown'
        else:
            if post_id in tweet_id_state_dict:
                location = '{}, United States'.format(state_code_name_dict[tweet_id_state_dict[post_id]])
            else:
                location = tweet_id_country_dict[post_id]

        tweet_contents.append((post_id, result[1], result[2], location, result[3], result[4], result[5]))
    return tweet_contents


topn = 100

top_tweets = []
for topic_id in range(NUM_TOPICS):
    top_tweets += [(topic_id, *tweet_info) for tweet_info in sorted(get_topn_tweets_for_topic(topic_id, topn), key=lambda tup: tup[1], reverse=True)]

df = pd.DataFrame(top_tweets, columns=['topic_id', 'post_id', 'fit_value', 'author_name', 'location', 'date', 'url', 'content'])
df.to_csv('{}/top{}_tweets.csv'.format(twitter_model_path, topn), encoding='utf-8-sig')

# top words

topn = 50

top_words = []
for topic_id in range(NUM_TOPICS):
    topic_top_words = model.get_topic_terms(topic_id, topn=topn)
    topic_top_words = [(model_dict[word_id], prob) for (word_id, prob) in topic_top_words]
    topic_top_words = [(topic_id, *tup) for tup in topic_top_words]
    top_words += topic_top_words

df = pd.DataFrame(top_words, columns=['topic_id', 'word', 'fit_value'])
df.to_csv('{}/top_terms.csv'.format(twitter_model_path), encoding='utf-8')

if not os.path.exists(twitter_model_path):
    os.makedirs(twitter_model_path)

T = time.time()

if not os.path.isfile('{}/tweet_date_map.json'.format(twitter_model_path)):
    #conn = sql.connect(database_path)
    cur = conn.cursor()
    query = 'SELECT post_id, date FROM posts'
    tweet_id_dates = cur.execute(query).fetchall()
    tweet_id_date_dict = {tweet_id : date for (tweet_id, date) in tweet_id_dates if str(tweet_id) in tweet_id_bow_dict}
    with open('{}/tweet_date_map.json'.format(twitter_model_path), 'w') as file_handle:
        json.dump(tweet_id_date_dict, file_handle)
else:
    with open('{}/tweet_date_map.json'.format(twitter_model_path), 'r') as file_handle:
        tweet_id_date_dict = json.load(file_handle)


T = time.time() - T
print('Loaded id -> date dictionary in {} seconds'.format(T), flush=True)


tweet_id_date_dict = {tweet_id : datetime.datetime.strptime(date_str[:date_str.index(' Jerusalem')], '%Y-%m-%d %H:%M:%S').astimezone(pytz.utc) for (tweet_id, date_str) in tweet_id_date_dict.items()}
start_date = min([date for (_, date) in tweet_id_date_dict.items()]).date()
end_date = max([date for (_, date) in tweet_id_date_dict.items()]).date()


def get_topn_most_popular(word_counts, topn):
    tuple_list = list(word_counts.items())
    sorted_list = sorted(tuple_list, key=lambda tup: tup[1], reverse=True)
    return sorted_list[:topn]


all_word_counts_per_day = []


def get_topn_words_per_day(start_date, end_date, topn):
    topn_most_popular_words_per_day = []

    current_date = start_date
    while start_date <= current_date <= end_date:
        date_word_counts = {}
        for tweet_id in tweet_id_date_dict:
            if tweet_id_date_dict[tweet_id].date() == current_date:
                for word_id_count_list in tweet_id_bow_dict[str(tweet_id)]:
                    word_id = word_id_count_list[0]
                    count = word_id_count_list[1]
                    word = model_dict[word_id]
                    date_word_counts[word] = date_word_counts.get(word, 0) + count
        all_word_counts_per_day.append(date_word_counts)
        topn_most_popular_words_per_day.append(get_topn_most_popular(date_word_counts, topn))

        current_date += datetime.timedelta(days=1)
    return topn_most_popular_words_per_day


def get_flattened_list(l):
    return [item for sublist in l for item in sublist]


topn = 50
most_popular_words_per_day = get_topn_words_per_day(start_date, end_date, topn)

flattened_list = get_flattened_list(most_popular_words_per_day)
index = []
for i in range(len(flattened_list)):
    if i % topn == 0:
        index.append(start_date + datetime.timedelta(days=int(i / topn)))
    else:
        index.append('')
pd.DataFrame(flattened_list, columns=['word', 'count'],
             index=index).to_csv('{}/top{}_words_daily.csv'.format(twitter_model_path, topn), encoding='utf-8-sig')

from collections import Counter


def sum_dictionary_values(dict_list):
    return dict(sum((Counter(dict(x)) for x in dict_list), Counter()))


# weekly

divided_list = [all_word_counts_per_day[x:x + 7] for x in range(0, (end_date - start_date).days + 1, 7)]
weekly_counts = []
for week in divided_list:
    weekly_counts.append(sum_dictionary_values(week))

most_popular_words_per_week = [get_topn_most_popular(weekly, topn) for weekly in weekly_counts]

flattened_list = get_flattened_list(most_popular_words_per_week)
index = []
for i in range(len(flattened_list)):
    if i % topn == 0:
        index.append(start_date + datetime.timedelta(days=7 * int(i / topn)))
    else:
        index.append('')
pd.DataFrame(flattened_list, columns=['word', 'count'],
             index=index).to_csv('{}/top{}_words_weekly.csv'.format(twitter_model_path, topn), encoding='utf-8-sig')

# monthly

monthly_counts = []
prev_month_number = start_date.month
daily_dicts_for_month = []
for i, daily_words in enumerate(all_word_counts_per_day):
    current_date = start_date + datetime.timedelta(days=i)
    month_number = current_date.month
    if month_number != prev_month_number:
        monthly_counts.append(sum_dictionary_values(daily_dicts_for_month))
        daily_dicts_for_month = []
        prev_month_number = month_number
    else:
        daily_dicts_for_month.append(daily_words)

most_popular_words_per_month = []
for month in monthly_counts:
    most_popular_words_per_month.append(get_topn_most_popular(month, topn))

import calendar

flattened_list = get_flattened_list(most_popular_words_per_month)
index = []
for i in range(len(flattened_list)):
    if i % topn == 0:
        month_num = start_date.month + (int(i / topn) % 12)
        if month_num > 12:
            month_num %= 12
        index.append(calendar.month_name[month_num])
    else:
        index.append('')
pd.DataFrame(flattened_list, columns=['word', 'count'],
             index=index).to_csv('{}/top{}_words_monthly.csv'.format(twitter_model_path, topn), encoding='utf-8-sig')