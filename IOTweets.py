import pandas as pd
from collections import Counter
from datetime import datetime
from nltk.tokenize import TweetTokenizer
import csv

#----------------------------build_df(filepath)------------------------------------
#build a dataframe containing columns "word" and "occurence" from a vocabulary.
# filepath : the path of the file in which the vocabulary is written


def build_df(filepath, bitri):

    tknzr = TweetTokenizer(preserve_case=False)

    #load the vocabulary
    df = pd.read_table(filepath_or_buffer = filepath, encoding="utf-8",  header=None, names=["word"])

    #remove blank space words (pollution)
    df["len"] = df["word"].map(lambda x : len(tokenize(tknzr, x)))
    if bitri:
        df = df[df["len"]>=2]
    else:
        df = df[df["len"]==2]
    df = df.drop(labels=["len"], axis = 1)

    #build the dataframe
    if bitri:
        df["occurence"] = df["word"].map(lambda x:  tokenize(tknzr, x)[0])
        df["word"] = df["word"].map(lambda x:  tuple(tokenize(tknzr, x)[1:]))
        return df
    else :
        df["occurence"] = df["word"].map(lambda x:  tokenize(tknzr, x)[0])
        df["word"] = df["word"].map(lambda x:  str(tokenize(tknzr, x)[1]))
        return df

#----------------------------tokenize(tknzr, line)------------------------------------
# transforms a line into a list of its tokens
# tknzr : the tokenizer to be used to tokenize
# line : the line to be tokenized
def tokenize(tknzr, line):
    tokens = tknzr.tokenize(line)
    tokens = [tok for t in tokens for tok in t.split()] # split at spaces and flatten
    return tokens

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


#----------------------------write_vocab_to_file(vocab_counter, dest_file_name)------------------------------------
#vocab_counter : the couter that counted the words
#dest_file_name : the name of the file in which vocab should be written

def write_vocab(tweets, cut_threshold, file_name , bitri):
    counter = build_vocab_counter(tweets, cut_threshold, bitri)
    write_vocab_to_file(counter, (file_name + "_cut=" +str(cut_threshold) +"_bitri="+str(bitri)))

#----------------------------write_relevance_to_file(relevant, dest_file_name)------------------------------------
#relevant : the dataframe of relevance that will be writen
#dest_file_name : the name of the file in which relevance should be written
def write_relevance_to_file(relevant, dest_file_name):
    with open(dest_file_name, "w", encoding="utf-8") as inputfile:
        for index, row in relevant.iterrows():
            inputfile.write(str(row["ratio"]))
            inputfile.write("\t")
            if type(row["word"]) is tuple:
                inputfile.write(" ".join(row["word"]))
            else:
                inputfile.write(str(row["word"]))
            inputfile.write("\n")


def write_index_to_file(vocab, dest_file_name):
    with open(dest_file_name, "w", encoding="utf-8") as inputfile:
        for index, row in relevant.iterrows():
            inputfile.write(str(row["index"]))
            inputfile.write("\t")
            if type(row["word"]) is tuple:
                inputfile.write(" ".join(row["word"]))
            else:
                inputfile.write(str(row["word"]))
            inputfile.write("\n")

#----------------------------extract_relevance(relevant_filename)------------------------------------
#returns the relevance etracted from a file
#relevant_filename : the name of the file that contains the relevance
def extract_relevance(relevant_filename):
    with open(relevant_filename, 'r', encoding="utf-8") as f:
        relevance = {}
        for line in f:
            split = line.split()
            token = tuple(split[1:])
            relevance[token] = split[0]
        return relevance


def extract_index(relevant_filename):
    with open(relevant_filename, 'r', encoding="utf-8") as f:
        relevance = {}
        for index, line in enumerate(f):
            split = line.split()
            token = tuple(index, split[1:][1])
            relevance[token] = split[0]
        return relevance
