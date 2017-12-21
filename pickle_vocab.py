#!/usr/bin/env python3
import pickle


def main( vocab_filename, dest_filename):
    vocab = dict()
    with open(vocab_filename) as f:
        for idx, line in enumerate(f):
            token = line.strip().split()[1:]
            if len(token) == 1:
                token = token[0]
            else:
                token = tuple(token)
            vocab[token] = idx

    with open(dest_filename, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
