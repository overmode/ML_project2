import pandas as pd
from datetime import datetime
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import bigrams, trigrams
from nltk.tokenize import TweetTokenizer

def merging(neg, pos, only_words = False):

    # We merge the two dataframe in order to better handle them
    merged = pd.merge(left=neg, right=pos, left_on = "word", right_on = "word", suffixes=('_neg', '_pos'),  how="outer")
    merged = merged.fillna(0)

    if(not only_words):
        #We only consider words whose occurences dfferences between sad and happy tweets is greater or equal than 5
        merged["difference"] = abs((merged["occurence_neg"]-merged["occurence_pos"]))
        merged = merged[merged["difference"]>=5]

        #We compute the sum of occurences
        merged["somme"] = merged["occurence_neg"]+merged["occurence_pos"]

        #The ratio si how relevant it is to judge happyness/sadness of the tweet using the word : 0 if not relevant, 1 if truly relevant
        merged["ratio"] = 2* abs(0.5 - merged["occurence_pos"]/(merged["occurence_pos"]+merged["occurence_neg"]))


    merged["len"] = merged["word"].map(len)

    #If we want to sort it
    #merged.sort_values(by = ["ratio","somme"], ascending=[False, False])

    return merged


def filter_single_rep(same_begin):
    cop = []

    for i, l in enumerate(same_begin):
        inner = []

        for j, w in enumerate(l):
            if "-" in w:
                continue
            inner.append(w)

        if len(inner)>1:
            cop.append(inner)

    return cop

def characteristic_words(merged_full, min_diff):

    mf_max_ratio = merged_full[(merged_full.ratio == 1) & (merged_full.difference >= min_diff)]
    mf_max_ratio = mf_max_ratio[["word","difference"]]
    word_max_ratio = list(mf_max_ratio.values)
    return word_max_ratio

def sem_by_repr(semantics, representative, tweet):
    """Retrun a tweet that countain only the representative of a given semantic """

    for semantic in semantics:
        if (semantic in tweet):
            tweet = tweet.replace(semantic,representative)
    return tweet

def sem_by_repr2(semantics, tweet):
    """Semanticss is a list of list of semantic where the first one is the representative.
    Retrun a tweet that countain only the representative of a given semantic below is an exemple of the semantics

    [['because', 'becausee'],
    ['someone', "someone's", 'someones', 'someonee'],
    ['friends', 'friendship', 'friendships', 'friendss', 'friendster']]
    """

    for semantic in semantics:
        representative = semantic[0]
        sem = semantic[1:]
        sem = sorted(sem, key =len, reverse= True)  # We sort by lenth on a non Ascending way to avoid the case where a
                                                    # String is a substring and would chang

        tweet = sem_by_repr(sem , representative, tweet)
    return tweet



def no_dot(tweet):
    tweet = tweet.split()
    for i, t in enumerate(tweet):
        if t == "...":
            continue
        if "." in t:
            if t[-1] == ".":
                t = t[:-1]

            t = t.replace("."," ")
            tweet[i] = t
    return " ".join(tweet)






def find_repetition(tweet):
    copy = ""
    i = 0
    while i<len(tweet):
        if (i<len(tweet)-3 and tweet[i] == tweet[i+1] and tweet[i] == tweet[i+2]):
            i = i+2
            while( i<len(tweet)-1 and tweet[i] == tweet[i+1] ):
                i+=1
            copy += tweet[i]
            i+=1
        else:
            copy += tweet[i]
            i+=1
    return copy

''' already done by nltk's stem
def no_s(tweet):
    tweet = tweet.split()
    for i, t in enumerate(tweet):

        if(len(t)>4 and t[-3:] == "ies"):
            tweet[i] = t[0:-3] + "y"

        if(len(t)>4 and t[-2:] == "es"):
            tweet[i] = t[0:-2] + "e"

        #if(len(t)>4 and t[-1]=="s"  and t[-2]!="s"):
        #    tweet[i] = t[0:-1]

    tweet = " ".join(tweet)
    return tweet
'''



def stem_tweet(tweet):
    ps = PorterStemmer()
    tweet = tweet.split()
    for i, t in enumerate(tweet):
        tweet[i] = ps.stem(t)

    return " ".join(tweet)

#We could improve the processed tweets by first f
def standardize_tweets(tweets):
    "filter and uniform the tweets "

    for i, tweet in enumerate(tweets):
        tweet = find_repetition(tweet)
        tweet = no_dot(tweet)
        #tweets[i] = no_s(tweet)

    return tweets

#We could improve the processed tweets by first f
def stem_tweets(tweets, semantic):
    "filter and uniform the tweets "

    for i, tweet in enumerate(tweets):
        tweet = sem_by_repr2(semantic, tweet)  #remplace haha
        tweets[i] = stem_tweet(tweet)

    return tweets


def token_to_string(token):
    return (str(token)).replace(" ", "")

def extract_tokens(tknzr, tweet_string):

    unigrams = [token for token in tknzr.tokenize(tweet_string)]

    def generator(unigrams):
        yield from unigrams
        yield from bigrams(unigrams)
        yield from trigrams(unigrams)

    return generator(unigrams)

def build_vocab_counter(tweets, cut_threshold, bitri):
    startTime= datetime.now()
    tknzr = TweetTokenizer(preserve_case=False)

    # add every stemmed tokens, bigrams and trigrams
    final_counter = Counter()
    loading_counter = 0
    for tweet in tweets:

        # add all tokens of a tweet in the counter
        if(bitri):
            final_counter.update(extract_tokens(tknzr, tweet))
        else :
            final_counter.update(tweet.split(" "))
        if loading_counter%1000==1:
                print("{:.1f}".format(loading_counter/len(tweets)*100), "%", end='\r')
        loading_counter+=1

    # remove less frequent items
    print("removing items present less than", cut_threshold, "times")
    final_set = Counter(token for token in final_counter.elements() if final_counter[token] >= cut_threshold)
    for k in list(final_counter):
        if final_counter[k] < cut_threshold:
            del final_counter[k]

    timeElapsed=datetime.now()-startTime

    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

    return final_counter


def add_bitri(tweets):
    "append bigrams and trigrams to the tweets "
    nb_tweets = len(tweets)
    for i, tweet in enumerate(tweets):
        to_store = ''
        for t in extract_tokens(tweet):
            to_store = to_store + " " + token_to_string(t)
        tweets[i] = to_store
        if i%1000==0:
            print("{:.1f}".format(i/nb_tweets*100), "%", end='\r')
    return tweets

def create_bitri_tweets(previous_name, dest, is_test_data):
    startTime= datetime.now()
    if is_test_data:
        tweets = import_without_comma(previous_name, is_test_data)
    else:
        tweets = import_without_comma(previous_name, is_test_data)
    tweets = add_bitri(tweets)
    export(tweets, dest)
    timeElapsed=datetime.now()-startTime
    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

#return true iff at least one of the tweets contains at least one of the words
#tweets is an array of strings
#words is a list of words
def contains(tweets, words):
    for t in tweets:
        for w1 in t.split():
            for w2 in words:
                if w1 == w2:
                    return True
    return False

def drop_duplicates(tweets):
    tweets = list(set(tweets))
    return tweets
