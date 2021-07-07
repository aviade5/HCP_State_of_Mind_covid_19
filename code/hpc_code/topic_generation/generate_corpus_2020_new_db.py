import sys
import pandas as pd
import sqlite3 as sql
import time
import random
from langdetect import detect
import re
from emoji import UNICODE_EMOJI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
import time
import json
import os

path = '/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
english_only = True

# read num_tweets random tweets from given path to database
# if number not given then read ALL tweets
def read_tweets(path, num_tweets=0, read_person_authors_only=False, sample_size=0):
    T = time.time()
    conn = sql.connect(path)
    cur = conn.cursor()
    query = "SELECT post_id, content FROM posts WHERE date > date('2019-12-31') {}".format(
        '' if num_tweets == 0 else 'ORDER BY RANDOM() LIMIT {}'.format(num_tweets))
    results = cur.execute(query)
    results = results.fetchall()

    print('Finished reading {} tweets in {} seconds'.format(len(results), time.time() - T), flush=True)
    if sample_size > 0:
        results = random.sample(results, sample_size)
        print('Generated a random sample of {} tweets'.format(sample_size), flush=True) 

    return {result[0]: result[1] for result in results}


def filter_tweets(tweet_id_dict):
    filtered_tweets = []
    bad_lang_tweets = []
    new_dict = {}

    non_english = 0
    none_tweets = 0

    for i, (tweet_id, tweet) in enumerate(tweet_id_dict.items()):

        if i % 50000 == 0:
            print('Finished filtering {}/{} tweets'.format(i, len(tweet_id_dict)), flush=True)

        if tweet is None:
          none_tweets += 1
          continue

        # REMOVE RE-TWEETS
        # if tweet.startswith('RT @'):
        #  continue

        # REMOVE \r\n
        tweet = tweet.replace('\r\n', ' ')

        tweet = tweet.replace('\n', ' ')

        tweet = tweet.replace('\t', ' ')

        # REMOVE EMOJIS
        for ch in tweet:
            if ch in UNICODE_EMOJI:
                tweet = tweet.replace(str(ch), '')

        # REMOVE @,&
        for word in tweet.split():
            if word.startswith('@') or word.startswith('&'):
                tweet = tweet.replace(word, '')

        # REMOVE &amp
        tweet = tweet.replace('&amp', '')

        # REMOVE tweets without any letters
        if not re.search('[a-zA-Z]', tweet):
            continue

        tweet = tweet.strip()

        # if tweet is now empty, nevermind
        if not tweet:
            continue

        # REMOVE NON-ENGLISH
        if english_only:
            try:
                lang = detect(tweet)
            except Exception as e:
                bad_lang_tweets.append(tweet)
                lang = 'ERROR'
            if lang == 'en':
                new_dict[tweet_id] = tweet
            else:
                non_english += 1
        else:
            new_dict[tweet_id] = tweet

    print("Couldn't detect language in {} tweets (either URL only or non-english)".format(len(bad_lang_tweets)), flush=True)
    print('Filtered {} non-English tweets'.format(non_english), flush=True)

    return new_dict


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False



def nltk_preprocess(tweet_id_dict, lemmatizer_name, with_corona_keywords=False):
    keyword_list = []
    new_dict = {}

    if not with_corona_keywords:
        try:
            with open('corona_keywords.txt', 'r', encoding='utf-8') as file_handle:
                keyword_list = file_handle.readlines()
                keyword_list = [keyword.replace('\n', '') for keyword in keyword_list]
        except Exception as e:
            print("Couldn't open corona keywords file:")
            print(e)
            print('Keywords WERE NOT REMOVED')

    stop_words = set(stopwords.words('english'))
    preprocessed_texts = []
    for i, (tweet_id, tweet) in enumerate(tweet_id_dict.items()):
        if i % 50000 == 0:
            print('Finished pre-processing {}/{} tweets'.format(i, len(tweet_id_dict)), flush=True)
        tokens = word_tokenize(tweet)
        new_text = []
        for word in tokens:
            word = word.lower()
            # remove stopwords, keywords, keep words that are alphanumeric, length greater than 2
            if not with_corona_keywords:
                if word not in stop_words and word not in keyword_list and word.isalnum() and len(
                        word) > 2 and word != 'http' and word != 'https' and 'covid' not in word:
                    word = word.replace('coronavirus', '')
                    new_text.append(word)
            else:
                if word not in stop_words and word.isalnum() and len(
                        word) > 2 and word != 'http' and word != 'https':
                    new_text.append(word)

        # Lemmatize

        # spacy lemmatization
        if lemmatizer_name == 'spacy':
            lemmatized = nlp(' '.join(new_text))
            new_text = [token.lemma_.lower() for token in lemmatized]

        # nltk lemmatization
        elif lemmatizer_name == 'nltk':
            lemmatizer = WordNetLemmatizer()
            new_text = [lemmatizer.lemmatize(w) for w in new_text]

        else:
            print('BAD LEMMATIZER!')

        if new_text:
            preprocessed_texts.append(new_text)
            new_dict[tweet_id] = new_text

    return new_dict

specified_sample_size = 0

if len(sys.argv) > 1 and sys.argv[1] == '-sample':
    folder_name = '70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS_SAMPLE{}'.format(int(sys.argv[2]))
    if sys.argv[3] != '-sample_size':
        print('Please specify sample size with -sample_size.', flush=True)
        raise ValueError('No sample size specified')
    else:
        specified_sample_size = int(sys.argv[4])
else:
    folder_name = '70_PERCENT_CONFIDENCE_53K_INDIVIDUAL_HCP_AUTHORS_2020_NO_KEYWORDS_WITH_RETWEETS'

corpus_directory = '/gpfs0/rami/users/iliapl/data/output_data/lda_corpora/{}'.format(folder_name)
num_tweets = 0

if not os.path.exists(corpus_directory):
    os.makedirs(corpus_directory)

tweet_id_dict = {}

preprocess_start = time.time()
tweet_list = filter_tweets(read_tweets(path, num_tweets, sample_size=specified_sample_size))
preprocessed_tweets = nltk_preprocess(tweet_list, 'nltk', with_corona_keywords=False)
print('{} tweets after pre-processing'.format(len(preprocessed_tweets)), flush=True)

# create dictionary for gensim
model_dict = Dictionary([content for (tweet_id, content) in preprocessed_tweets.items()])

# create corpus for gensim (word -> frequency)
corpus = []
for tweet_id, doc in preprocessed_tweets.items():
    bow = model_dict.doc2bow(doc)
    corpus.append(bow)
    tweet_id_dict[tweet_id] = bow

model_dict.save('{}/dict.id2word'.format(corpus_directory))
MmCorpus.serialize('{}/corpus.mm'.format(corpus_directory), corpus, id2word=model_dict)
with open('{}/post_id_bow_dict.json'.format(corpus_directory), 'w') as json_file:
    json.dump(tweet_id_dict, json_file)

print('pre-processing took {} seconds'.format(time.time() - preprocess_start), flush=True)