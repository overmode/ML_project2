import pandas as pd
from collections import Counter
from datetime import datetime
from ProcessTweets import *

def build_df(filepath):
    """from a cut_vocab return a dataframe which is a mapping of words in tweets
    with their occurences in all tweets
    take the path of the file of the tweets
    """

    df = pd.read_table(filepath_or_buffer = filepath, header=None, names=["word"])
    df["occurence"] = df["word"].map(lambda x:  int(x.split()[0]))
    df["word"] = df["word"].map(lambda x:  x.split()[1])
    return df

def import_(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip() for line in f]
    return tweets

def import_without_comma(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip()[line.find(",")+1:] for line in f]     # Make sure to withdraw the "nbr",
    return tweets

def export(tweets, name):
    with open(name, 'w', encoding="utf-8") as f:
        f.write("\n".join(tweets))


def build_vocab(tweets, dest_file_name, cut):

    # add stemmed tokens, bigrams and trigrams
    final_set = Counter()
    startTime= datetime.now()
    len_tweets = len(tweets)
    counter = 0
    for tweet in tweets:
        # add all tokens of a tweet in the counter
        final_set.update(tweet)
        if counter%1000==1:
            print("{:.1f}".format(counter/len_tweets*100), "%", end='\r')
        counter+=1

    # remove less frequent items
    print("removing items present less than", cut, "times")
    final_set = Counter(token for token in final_set.elements() if final_set[token] >= cut)

    with open(dest_file_name, "w") as inputfile:
        for token, count in final_set.most_common():
            inputfile.write(str(count))
            inputfile.write(" ")
            inputfile.write(str(token))
            inputfile.write("\n")
    timeElapsed = datetime.now() - startTime
    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
