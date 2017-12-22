#!/usr/bin/env python3
import pickle


def to_vocab_pickle(path_vocab_full= 'vocab_full', path_vocab_dest = 'vocab_full.pkl'):
    """
    path_vocab is the vocabe that we want to pickle.
    this vocabe should countain all the voacab present in tweet pos and neg.
    Return Nothing but writing on the file path_vocab_dest
    """

    vocab = dict()
    with open(path_vocab_full) as f:
        for idx, line in enumerate(f):
            token = line.strip().split()[1:]
            if len(token) == 1:
                token = token[0]
            else:
                token = tuple(token)
            vocab[token] = idx

    with open('vocab_full.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
