from nltk import bigrams, trigrams
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(preserve_case=False)
ps = PorterStemmer()


def extract_tokens(tweet_string):

    unigrams = [ps.stem(token) for token in tknzr.tokenize(tweet_string)]

    def generator(unigrams):
        yield from unigrams
        yield from bigrams(unigrams)
        yield from trigrams(unigrams)

    return generator(unigrams)
