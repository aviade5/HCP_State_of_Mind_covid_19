import gensim
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import LdaModel, LdaMulticore
import time
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import json
import logging
import re
import operator
import sys

NUM_CORES = 24

corpus_path = r'/gpfs0/rami/users/iliapl/data/output_data/lda_corpora/{}'.format(sys.argv[1])
NUM_TOPICS = int(sys.argv[2])

output_path = r'/gpfs0/rami/users/iliapl/data/output_data/lda_models/{}_{}_TOPICS'.format(sys.argv[1], NUM_TOPICS)


if not os.path.exists(output_path):
    os.makedirs(output_path)


def train_lda(corpus, model_dict, num_topics, num_passes, chunksize, eval_every):
    return LdaMulticore(
                            corpus=corpus,
                            id2word=model_dict,
                            random_state=100,
                            num_topics=num_topics,
                            passes=num_passes,
                            chunksize=chunksize,
                            batch=False,
                            alpha='asymmetric',
                            decay=0.5,
                            offset=64,
                            eta=None,
                            eval_every=eval_every,
                            iterations=100,
                            gamma_threshold=0.001,
                            per_word_topics=True,
                            workers=NUM_CORES)


def save_convergence_plot(log_dir):
    p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open('{}/training.log'.format(log_dir))]
    matches = [m for m in matches if len(m) > 0]
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    liklihood = [float(t[0]) for t in tuples]
    iter = list(range(0, len(tuples) * 10, 10))
    plt.plot(iter, liklihood, c='black')
    plt.ylabel('log liklihood')
    plt.xlabel('iteration')
    plt.title('Topic Model Convergence')
    plt.grid()
    plt.savefig('{}/convergence_likihood.png'.format(log_dir))
    plt.close()


def generate_word_cloud(model_dict, lda_model, num_topics, directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for topic in range(0, num_topics):
        topic_terms = lda_model.get_topic_terms(topic, 20)
        word_frequency_dict = {model_dict[word_id]: freq for (word_id, freq) in topic_terms}

        # make wordcloud
        wc = WordCloud(background_color='black', width=1000, height=1000, max_words=20).generate_from_frequencies(
            word_frequency_dict)
        wc.to_file('{}/topic{}.png'.format(directory, topic))


def get_document_topics(corpus, lda):
    topics = []
    for i, document in enumerate(corpus):
        corpus_dict = {topic:prob for (topic, prob) in lda.get_document_topics(corpus[i])}
        topics.append(max(corpus_dict.items(), key=operator.itemgetter(1))[0])
    return topics


def detect_topics(output_path, corpus, model_dict, num_topics, num_passes, chunksize=15000, eval_every=10,
                  print_results=False,
                  save_wordclouds=False, hebrew=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename='{}/training.log'.format(output_path), format='%(asctime)s:%(levelname)s:%(message)s',
                        level=logging.INFO)

    T = time.time()
    lda_model = train_lda(corpus, model_dict, num_topics, num_passes, chunksize, eval_every)
    T = time.time() - T

    print('Finished {} topics and {} passes ({} seconds)'.format(num_topics, num_passes, T), flush=True)
    if print_results:
        print(lda_model.print_topics(-1), flush=True)
    lda_model.save('{}/lda.model'.format(output_path))
    print('Saved to {}/lda.model'.format(output_path), flush=True)
    save_convergence_plot(output_path)

    if save_wordclouds:
        for topic in range(0, num_topics):
            topic_terms = lda_model.get_topic_terms(topic, 20)
            if not hebrew:
                word_frequency_dict = {model_dict[word_id]: freq for (word_id, freq) in topic_terms}
            else:
                word_frequency_dict = {get_display(model_dict[word_id]): freq for (word_id, freq) in topic_terms}

            # make wordcloud
            if not hebrew:
                wc = WordCloud(background_color='black', width=1000, height=1000,
                               max_words=20).generate_from_frequencies(word_frequency_dict)
            else:
                wc = WordCloud(background_color='black', width=1000, height=1000, max_words=20,
                               font_path=r'C:\WINDOWS\FONTS\AHRONBD.TTF').generate_from_frequencies(word_frequency_dict)
            wc.to_file('{}/topic{}.png'.format(output_path, topic))

    return lda_model


def get_all_topic_term_ids(lda_model, num_topics, num_terms):
    terms = []
    for topic in range(0, num_topics):
        terms.append([word for (word, freq) in lda_model.get_topic_terms(topic, num_terms)])
    return terms


def get_all_topic_terms(model_dict, lda_model, num_topics, num_terms):
    terms = []
    for topic in range(0, num_topics):
        terms.append([model_dict[word] for (word, freq) in lda_model.get_topic_terms(topic, num_terms)])
    return terms


corpus = MmCorpus('{}/corpus.mm'.format(corpus_path))
model_dict = Dictionary.load('{}/dict.id2word'.format(corpus_path))

model = detect_topics(output_path, corpus, model_dict, NUM_TOPICS, 150, eval_every=0, chunksize=130000, print_results=False, save_wordclouds=True, hebrew=False)
