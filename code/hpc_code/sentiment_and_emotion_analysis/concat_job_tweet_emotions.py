import pandas as pd

DATA_PATH = '/gpfs0/rami/users/iliapl/data/emotion_recognition'
FILE_NAME = 'job{}_tweets_with_emotions.csv'
NUM_JOBS = 5

dfs = []
for i in range(NUM_JOBS):
  dfs.append(pd.read_csv('{}/{}'.format(DATA_PATH, FILE_NAME.format(i + 1))))
  
print('Finished appending dataframes. Saving to {}'.format(DATA_PATH), flush=True)
  
pd.concat(dfs, ignore_index=True).to_csv('{}/all_tweet_emotions.csv'.format(DATA_PATH), index=False)