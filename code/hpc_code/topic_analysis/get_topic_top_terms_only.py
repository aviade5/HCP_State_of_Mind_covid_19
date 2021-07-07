import sys
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pandas as pd

twitter_model_path = sys.argv[1]
NUM_TOPICS = int(sys.argv[2])

topn = int(sys.argv[3])

model_dict = Dictionary.load('{}/dict.id2word'.format(twitter_model_path))
model = LdaModel.load('{}/lda.model'.format(twitter_model_path))

top_words = {}
for topic_id in range(NUM_TOPICS):
    topic_top_words = model.get_topic_terms(topic_id, topn=topn)
    topic_top_words = [model_dict[word_id] for (word_id, prob) in topic_top_words]
    top_words['topic{}'.format(topic_id)] = topic_top_words

df = pd.DataFrame(top_words)
df.to_csv('{}/top_{}_terms.csv'.format(twitter_model_path, topn), encoding='utf-8', index=False)