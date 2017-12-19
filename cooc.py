#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from nltk import bigrams, trigrams


def main():
    with open('vocab_full.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in ['preprocessed_pos', 'preprocessed_neg']:
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
    with open('cooc_full.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
