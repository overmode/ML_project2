import pandas as pd
from collections import Counter
from datetime import datetime
from ProcessTweets import *
import csv

def build_df(filepath):
    """from a cut_vocab return a dataframe which is a mapping of words in tweets
    with their occurences in all tweets
    take the path of the file of the tweets
    """


    df = pd.read_table(filepath_or_buffer = filepath, encoding="utf-8",  header=None, names=["word"])
    df["len"] = df["word"].map(lambda x : len(x.split()))
    df = df[df["len"]==2]
    df = df.drop(labels=["len"], axis = 1)
    df["occurence"] = df["word"].map(lambda x:  x.split()[0])
    df["word"] = df["word"].map(lambda x:  x.split()[1])
    return df

def import_(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip() for line in f]
    return tweets

def import_without_id(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip()[line.find(",")+1:] for line in f]     # Make sure to withdraw the "nbr",
    return tweets

def export(tweets, name):
    with open(name, 'w', encoding="utf-8") as f:
        f.write("\n".join(tweets))

def write_vocab_to_file(vocab_counter, dest_file_name):
    with open(dest_file_name, "w", encoding="utf-8") as inputfile:
        for token, count in vocab_counter.most_common():
            inputfile.write(str(count))
            inputfile.write(" ")
            if type(token) is tuple:
                inputfile.write(" ".join(token))
            else:
                inputfile.write(str(token))
            inputfile.write("\n")
