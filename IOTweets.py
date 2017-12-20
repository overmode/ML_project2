import pandas as pd
from collections import Counter
from datetime import datetime
from nltk.tokenize import TweetTokenizer
import csv

#----------------------------build_df(filepath)------------------------------------
#build a dataframe containing columns "word" and "occurence" from a vocabulary.
# filepath : the path of the file in which the vocabulary is written


<<<<<<< HEAD
def build_df(filepath):
=======
def build_df(filepath, bitri):
>>>>>>> origin/master
    
    tknzr = TweetTokenizer(preserve_case=False)

    #load the vocabulary
    df = pd.read_table(filepath_or_buffer = filepath, encoding="utf-8",  header=None, names=["word"])

    #remove blank space words (pollution)
    df["len"] = df["word"].map(lambda x : len(tknzr.tokenize(x)))
    df = df[df["len"]>=2]
    df = df.drop(labels=["len"], axis = 1)

    #build the dataframe
<<<<<<< HEAD
    df["occurence"] = df["word"].map(lambda x:  tknzr.tokenize(x)[0])
    df["word"] = df["word"].map(lambda x:  tuple(tknzr.tokenize(x)[1:]))
    return df
=======
    if bitri:
        df["occurence"] = df["word"].map(lambda x:  int(tknzr.tokenize(x)[0]))
        df["word"] = df["word"].map(lambda x:  tuple(tknzr.tokenize(x)[1:]))
        return df
    else :
        df["occurence"] = df["word"].map(lambda x:  int(tknzr.tokenize(x)[0]))
        df["word"] = df["word"].map(lambda x:  str(tknzr.tokenize(x)[1]))
        return df

#----------------------------import_(path)------------------------------------
#import tweets written in file stocked under given path as an array of tweets (string)
# path : the path of the file in which the tweets are written
>>>>>>> origin/master

#----------------------------import_(path)------------------------------------
#import tweets written in file stocked under given path as an array of tweets (string)
# path : the path of the file in which the tweets are written

def import_(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip() for line in f]
    return tweets

#----------------------------import_without_id(path)------------------------------------
#import tweets written in file stocked under given path as an array of tweets (string), but remove the id in file
# path : the path of the file in which the tweets are written

def import_without_id(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip()[line.find(",")+1:] for line in f]     # Make sure to withdraw the "nbr",
    return tweets

#----------------------------export(tweets, name)------------------------------------
#export an array of tweets (strings) in a file which name is given
#tweets : the tweets that we want to export (an array of strings)
#name : the name of the file in which tweets should be written

def export(tweets, name):
    with open(name, 'w', encoding="utf-8") as f:
        f.write("\n".join(tweets))


#----------------------------write_vocab_to_file(vocab_counter, dest_file_name)------------------------------------
#export the count of a given counter in a file which name is dest_file_name
#vocab_counter : the counter that contains words and their occurences
#dest_file_name : the name of the file in which vocabulary should be written


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
