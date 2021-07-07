import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATABASE_PATH = '/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db'
EMOTIONS_PATH = '/gpfs0/rami/users/iliapl/data/emotion_recognition'

labels = ['Anger',
          'Disgust',
          'Fear',
          'Joy',
          'Sadness',
          'Surprise']
colors = ['red', 
          'brown', 
          'orange', 
          'limegreen', 
          'grey', 
          'deepskyblue']

con = sql.connect(DATABASE_PATH)

tweet_dates_df = pd.read_sql('SELECT post_id, date FROM posts', con)

all_emotions_df = pd.read_csv('{}/all_tweet_emotions.csv'.format(EMOTIONS_PATH))

# Separate between 2019 and 2020 tweets

separating_date = '2020-04-01'

all_emotions_2019_df = all_emotions_df[all_emotions_df['post_id'].isin(tweet_dates_df[tweet_dates_df['date'] < separating_date]['post_id'])]

print('Got {} tweets for 2019'.format(len(all_emotions_2019_df)), flush=True)

all_emotions_2020_df = all_emotions_df[all_emotions_df['post_id'].isin(tweet_dates_df[tweet_dates_df['date'] >= separating_date]['post_id'])]

print('Got {} tweets for 2020'.format(len(all_emotions_2020_df)), flush=True)

          
before_pandemic_aggregated_emotions = {label: all_emotions_2019_df[label].sum() for label in labels}
during_pandemic_aggregated_emotions = {label: all_emotions_2020_df[label].sum() for label in labels}

for emotion in labels:
    before_pandemic_aggregated_emotions[emotion] /= len(all_emotions_2019_df)
    during_pandemic_aggregated_emotions[emotion] /= len(all_emotions_2020_df)

print(before_pandemic_aggregated_emotions, flush=True)
print(during_pandemic_aggregated_emotions, flush=True)

f, ax = plt.subplots(1, 2, figsize=(15, 8))

mpl.rcParams['font.size'] = 15.0

ax[0].pie(before_pandemic_aggregated_emotions.values(), colors=colors, autopct='%1.1f%%', radius=1.2)
ax[0].set_title('Before Pandemic')

ax[1].pie(during_pandemic_aggregated_emotions.values(), colors=colors, autopct='%1.1f%%', radius=1.2)
ax[1].set_title('During Pandemic')

plt.legend(labels, loc='upper center', bbox_to_anchor=(-0.1, -0.1), fancybox=True, shadow=True, ncol=len(labels))

plt.savefig('{}/tweet_emotions_before_during_pandemic_separator_{}.png'.format(EMOTIONS_PATH, separating_date))