from nltk import bigrams, trigrams
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(preserve_case=False)


def extract_tokens(tweet_string):

    unigrams = [token for token in tknzr.tokenize(tweet_string)]

    def generator(unigrams):
        yield from unigrams
        yield from bigrams(unigrams)
        yield from trigrams(unigrams)

    return generator(unigrams)
