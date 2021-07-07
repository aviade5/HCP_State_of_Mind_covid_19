import time
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import LdaModel
import json
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami
import sys
import numpy as np
from sklearn import metrics

def get_bow_topic_id(bow, model):
    return sorted(model.get_document_topics(bow), key=lambda tup: tup[1], reverse=True)[0][0]
    

def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
model1_path = sys.argv[1]
model2_path = sys.argv[2]

print('Getting clustering quality for:', flush=True)
print(model1_path, flush=True)
print(model2_path, flush=True)

T = time.time()

model1 = LdaModel.load(f'{model1_path}/lda.model')
model2 = LdaModel.load(f'{model2_path}/lda.model')

with open(f'{model1_path}/post_id_bow_dict.json', 'r') as f:
    model1_id_bow_dict = json.load(f)
    
with open(f'{model2_path}/post_id_bow_dict.json', 'r') as f:
    model2_id_bow_dict = json.load(f)

common_tweets = list(set(model1_id_bow_dict.keys()) & set(model2_id_bow_dict.keys()))

print(f'Loaded models in {time.time() - T} seconds', flush=True)

print(f'Model 1 # tweets: {len(model1_id_bow_dict)}, model 2 # tweets: {len(model2_id_bow_dict)}, # common tweets: {len(common_tweets)}', flush=True)


T = time.time()

model1_tweet_topics = [get_bow_topic_id(model1_id_bow_dict[tweet], model1) for tweet in common_tweets]
model2_tweet_topics = [get_bow_topic_id(model2_id_bow_dict[tweet], model2) for tweet in common_tweets]

print(f'Created inputs in {time.time() - T} seconds', flush=True)

print(f'Purity: {purity_score(model1_tweet_topics, model2_tweet_topics)}', flush=True)
print(f'NMI: {nmi(model1_tweet_topics, model2_tweet_topics)}', flush=True)
print(f'ARI: {ari(model1_tweet_topics, model2_tweet_topics)}', flush=True)