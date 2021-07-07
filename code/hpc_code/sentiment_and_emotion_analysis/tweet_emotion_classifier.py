import json
import time
from emotion_predictor import EmotionPredictor
from keras import backend as K
import os
from importlib import reload
import random
import sys
import math
import pandas as pd
import numpy as np

BATCH_NUMBER = int(sys.argv[1])

BATCH_PATH = './tweets_to_classify/job{}_tweets.csv'.format(BATCH_NUMBER)
OUTPUT_FILE_PATH = '/gpfs0/rami/users/iliapl/data/emotion_recognition'

if not os.path.exists(OUTPUT_FILE_PATH):
    os.makedirs(OUTPUT_FILE_PATH)


# load emotion model
def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


set_keras_backend('theano')

T = time.time()

model = EmotionPredictor(classification='ekman', setting='mc')

print('Loaded emotion predictor model in {} seconds'.format(time.time() - T), flush=True)

T = time.time()

batch_df = pd.read_csv(BATCH_PATH)

print('Loaded batch tweets in {} seconds'.format(time.time() - T), flush=True)

mini_batches = np.array_split(batch_df, 5000)

result_df = pd.DataFrame()

for i, mini_batch in enumerate(mini_batches):
  T = time.time()
  probs = model.predict_probabilities(list(mini_batch['content']))
  mini_batch_final_df = pd.concat([mini_batch.reset_index(drop=True), probs.drop(['Tweet'], axis=1)], axis=1)
  result_df = pd.concat([result_df, mini_batch_final_df], ignore_index=True)
  print('Finished mini-batch {}/{} in {} seconds. Sleeping for 5 seconds...'.format(i, len(mini_batches), time.time() - T), flush=True)
  time.sleep(5)
  

T = time.time()

result_df.to_csv('{}/job{}_tweets_with_emotions.csv'.format(OUTPUT_FILE_PATH, BATCH_NUMBER), index=False)
