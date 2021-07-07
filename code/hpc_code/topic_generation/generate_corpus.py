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

path = '/gpfs0/rami/users/iliapl/data/databases/bad_actors_Coronavirus_Project_POI_Followers_13-06-20.db'
english_only = True

labeled_authors_df = pd.read_csv('/gpfs0/rami/users/iliapl/data/databases/person_organization_classification/labeled_authors_V10.csv')
unlabeled_authors_df = pd.read_csv('/gpfs0/rami/users/iliapl/data/databases/person_organization_classification/all_unlabeled_predictions_using_description_and_SVM_classifier_inbalanced_to_label_V10.csv')

person_authors = set(labeled_authors_df[labeled_authors_df['author_sub_type'] == 'PERSON']['author_screen_name'])

# get X% top confidence authors

CONFIDENCE_PERCENTILE = 0.4

person_sorted_by_confidence = unlabeled_authors_df[unlabeled_authors_df['str_automatic_prediction'] == 'PERSON'].sort_values(by=['confidence_to_organization'], ascending=True)
max_organization_confidence = person_sorted_by_confidence.quantile(CONFIDENCE_PERCENTILE)[0]
percentile_person_authors = person_sorted_by_confidence[unlabeled_authors_df['confidence_to_organization'] < max_organization_confidence]

person_authors |= set(percentile_person_authors['author_screen_name'])

print(f'Minimum confidence for {len(person_authors)} authors: {1 - percentile_person_authors.iloc[-1]["confidence_to_organization"]}', flush=True)


# get all authors with confidence > X%

#MIN_CONFIDENCE = 0.9

#person_authors |= set(unlabeled_authors_df[unlabeled_authors_df['confidence_to_organization'] < (1 - MIN_CONFIDENCE)]['author_screen_name'])



# read num_tweets random tweets from given path to database
# if number not given then read ALL tweets
def read_tweets(path, num_tweets=0, read_person_authors_only=False, sample_size=0):
    T = time.time()
    conn = sql.connect(path)
    cur = conn.cursor()
    if not read_person_authors_only:
        query = "SELECT post_id, content FROM posts WHERE date > date('2019-12-31') {}".format(
            '' if num_tweets == 0 else 'ORDER BY RANDOM() LIMIT {}'.format(num_tweets))
        results = cur.execute(query)
        results = results.fetchall()
        final_results = results
    else:
        query = "SELECT post_id, author, content FROM posts WHERE date > date('2019-12-31') {}".format(
            '' if num_tweets == 0 else 'ORDER BY RANDOM() LIMIT {}'.format(num_tweets))
        results = cur.execute(query)
        results = results.fetchall()
        final_results = []

        for post_id, author, content in results:
            if author in person_authors:
                final_results.append((post_id, content))

    print('Finished reading {} tweets in {} seconds'.format(len(final_results), time.time() - T), flush=True)
    if sample_size > 0:
        final_results = random.sample(final_results, sample_size)

    if sample_size > 0: print('Generated a random sample of {} tweets'.format(sample_size), flush=True)
    return {result[0]: result[1] for result in final_results}


def filter_tweets(tweet_id_dict):
    filtered_tweets = []
    bad_lang_tweets = []
    new_dict = {}

    non_english = 0

    for i, (tweet_id, tweet) in enumerate(tweet_id_dict.items()):

        if i % 50000 == 0:
            print('Finished filtering {}/{} tweets'.format(i, len(tweet_id_dict)), flush=True)

        # REMOVE RE-TWEETS
        if tweet.startswith('RT @'):
          continue

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


#import en_core_web_sm

#nlp = en_core_web_sm.load()


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
    for tweet_id, tweet in tweet_id_dict.items():
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

#folder_name = 'POI_Followers_13-06-20_PERSON_ONLY_V10_TOP40PERCENT_WITH_KEYWORDS_NO_RETWEETS'
folder_name = 'POI_Followers_13-06-20_PERSON_ONLY_V10_TOP40PERCENT_WITH_KEYWORDS_NO_RETWEETS_SAMPLE{}'.format(int(sys.argv[1]))
corpus_directory = '/gpfs0/rami/users/iliapl/data/output_data/lda_corpora/{}'.format(folder_name)
num_tweets = 0

if not os.path.exists(corpus_directory):
    os.makedirs(corpus_directory)

tweet_id_dict = {}

preprocess_start = time.time()
tweet_list = filter_tweets(read_tweets(path, num_tweets, read_person_authors_only=True, sample_size=2400000))
preprocessed_tweets = nltk_preprocess(tweet_list, 'nltk', with_corona_keywords=True)
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