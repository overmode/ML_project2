#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from nltk import bigrams, trigrams


def to_cook(path_vocab = 'vocab_full.pkl', path_tweets_pos = "clean_pos_bitri=True", path_tweets_neg = "clean_neg_bitri=True",path_tweets_test ="clean_test_bitri=True" dest = 'cooc_full.pkl'):
    """This method takes optionally the path of the pickled vocab
    and the path of the tweets. It then build with them to co-ocurrence matrix
    write the result in the file dest
    """

    with open(path_vocab, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in [path_tweets_pos, path_tweets_neg, path_tweets_test]:
        with open(fn) as f:
            for line in f:
                tokens = line.strip().split()

                def generator(unigrams):
                    yield from unigrams
                    yield from bigrams(unigrams)
                    yield from trigrams(unigrams)

                tokens_ids = [vocab.get(t, -1) for t in generator(tokens)]
                tokens_ids = [t for t in tokens_ids if t >= 0]
                for t in tokens_ids:
                    for t2 in tokens_ids:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    print("creating sparse matrix")
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(dest, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
