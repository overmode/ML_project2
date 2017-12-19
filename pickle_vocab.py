#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open('preprocessed_vocab_pos') as f:
        for idx, line in enumerate(f):
            token = line.strip().split()[1:]
            if len(token) == 1:
                token = token[0]
            else:
                token = tuple(token)
            vocab[token] = idx

    with open('vocab_full.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
