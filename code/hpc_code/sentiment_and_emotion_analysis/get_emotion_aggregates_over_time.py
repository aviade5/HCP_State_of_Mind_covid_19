database_path = '/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
all_emotions_path = '/gpfs0/rami/users/iliapl/data/emotion_recognition/all_tweet_emotions.csv'
output_path = '/gpfs0/rami/users/iliapl/data/emotion_recognition'

import sqlite3 as sql
import pandas as pd
import time

con = sql.connect(database_path)

T = time.time()

tweet_dates = pd.read_sql('SELECT post_id, date FROM posts', con)

# if we don't do this, merging takes a very very long time
tweet_dates = tweet_dates[~tweet_dates['post_id'].isnull()]
tweet_dates['post_id'] = tweet_dates['post_id'].astype('int64')

print(f'Loaded tweet dates in {time.time() - T} seconds', flush=True)

T = time.time()

all_emotions_df = pd.read_csv(all_emotions_path)

# if we don't do this, merging takes a very very long time
all_emotions_df = all_emotions_df[~all_emotions_df['post_id'].isnull()]
all_emotions_df['post_id'] = all_emotions_df['post_id'].astype('int64')

print(f'Loaded emotions in {time.time() - T} seconds', flush=True)

T = time.time()

merged_df = tweet_dates.merge(all_emotions_df, on='post_id', how='inner')

print(f'Merged dataframes in {time.time() - T} seconds', flush=True)

#num_frames = 50
num_days_per_period = 7
#min_date = merged_df['date'].min()
#min_date = min_date[:min_date.index(' J')]
min_date = '2020-01-01 00:00:00'
max_date = merged_df['date'].max()
max_date = max_date[:max_date.index(' J')]

from datetime import datetime

def date_range(start, end, intv):
    start = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    diff = (end  - start) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%Y-%m-%d %H:%M:%S")
    yield end.strftime("%Y-%m-%d %H:%M:%S")
    
frame_separators = pd.date_range(min_date, max_date, freq='{}D'.format(num_days_per_period))
last_date = str(frame_separators[-1])

if last_date[:last_date.index(' ')] != max_date[:max_date.index(' ')]:
  frame_separators = frame_separators.insert(len(frame_separators), frame_separators[-1] + pd.Timedelta(days=num_days_per_period))

date_windows = [(frame_separators[i], frame_separators[i+1]) for i in range(len(frame_separators) - 1)]
date_windows = [(str(i), str(j)) for (i, j) in date_windows]
    
aggregate_emotions_df = pd.DataFrame(columns=['start_date', 'end_date', 'Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'])

T = time.time()

for i, (start_date, end_date) in enumerate(date_windows):
  if i < len(date_windows) - 1:
    window_tweets = merged_df[(merged_df['date'] >= start_date) & (merged_df['date'] < end_date)]
  else:
    window_tweets = merged_df[(merged_df['date'] >= start_date) & (merged_df['date'] <= end_date)]

  # get aggregate emotions for this window
  entry_to_append = {'start_date': start_date, 'end_date': end_date}
  for emotion in ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']:
    entry_to_append[emotion] = window_tweets[emotion].mean()

  aggregate_emotions_df = aggregate_emotions_df.append(entry_to_append, ignore_index=True)
  print('Finished {}/{} frames in {} seconds'.format(i + 1, len(date_windows), time.time() - T), flush=True)
    
    
FILE_NAME = 'aggregate_emotions_2020.csv'

aggregate_emotions_df.to_csv('{}/{}'.format(output_path, FILE_NAME), index=False)

print('Finished', flush=True)