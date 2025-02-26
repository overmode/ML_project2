import pandas as pd
from datetime import datetime
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import bigrams, trigrams
from nltk.tokenize import TweetTokenizer
from IOTweets import *

#----------------------------merging(neg, pos, only_words = False)------------------------------------
#This function merges the dataframes associated to positive and negative words.
# neg : the dataframe of negative words
# pos : the dataframe of positive words
#only_words : boolean, True if you only want word and occurences columns (used to merge test_data dataframe as well)

def merging(neg, pos, only_words = False):

    # We merge the two dataframe in order to better handle them
    merged = pd.merge(left=neg, right=pos, left_on = "word", right_on = "word", suffixes=('_neg', '_pos'),  how="outer")
    merged = merged.fillna(0)

    if(not only_words):

        #insure type int
        merged["occurence_neg"] = merged["occurence_neg"].map(int)
        merged["occurence_pos"] = merged["occurence_pos"].map(int)
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

#----------------------------filter_single_rep(same_begin)------------------------------------
#Avoid mapping of composed words onto roots (words with "-") and delete inner lists containng only root
#same_begin : A list of list of strings, each inner list's first element is a root on which following words of the inner lists should be mapped
def filter_single_rep(same_begin):
    cop = []

    for i, l in enumerate(same_begin):
        inner = []

        #do not map a word if it contains a '-' character
        for j, w in enumerate(l):
            if "-" in w:
                continue
            inner.append(w)

        #remove inner lists containing only roots (they are not useful)
        if len(inner)>1:
            cop.append(inner)

    return cop


#----------------------------create_relevant_vocab(pertinence_thres, min_count, dataframe)------------------------------------
#creates the relevant vocab of words contained in the dataframe
#dataframe : the dataframe containing the words that could be choses to be part of the vocab
#min_count : the minimum number of total occurences a word in the relevant dictionnary should contains
#pertinence_thres : the minimum pertinence that a word in the relevant dictionnary should be associated to

def create_relevant_vocab(pertinence_thres, min_count, dataframe):

    #Keep only words with enough pertinence
    relevant = dataframe[dataframe["ratio"] >= pertinence_thres]

    #keep only words with a sufficient number of total occurences
    relevant = relevant[(relevant["occurence_pos"] + relevant["occurence_neg"]) >= min_count]

    #construct the vocab and save it as a .txt file
    relevant = relevant[["ratio","word"]]
    relevant.set_index("word")
    write_relevance_to_file(relevant, "relevant_vocab_pert="+str(pertinence_thres)+"_count="+str(min_count))
    



#----------------------------clean_tweets(path_pos, path_neg, path_test, is_full, cut_threshold)------------------------------------
#remove dots and repetitions in tweets, export the preprocessed tweets and their associated vocabs
#path_pos : path towards the file containing pos tweets
#path_neg : path towards the file containing neg tweets
#path_test : path towards the file containing test tweets
#is_full : True iff we load the full version of tweets
#cut_threshold : min number of occurence that a word contained in the vocab should have


def clean_tweets(path_pos, path_neg, path_test, is_full, cut_threshold):

    print("Import Data")
    #import tweets
    tweets_test = import_without_id(path_test)
    tweets_pos =  import_(path_pos)
    tweets_neg =  import_(path_neg)

    #remove duplicates from training data
    tweets_pos = drop_duplicates(tweets_pos)
    tweets_neg = drop_duplicates(tweets_neg)

    #process tweets
    print("Process data pos")
    cleaned_pos = standardize_tweets( tweets_pos)
    print("Process data neg")
    cleaned_neg = standardize_tweets( tweets_neg) #Sous form d'un tableau de tweet
    print("Process data test")
    cleaned_test = standardize_tweets( tweets_test) #Sous form d'un tableau de tweet

    #build the counters
    print("build vocab data pos")
    vocab_pos = build_vocab_counter(cleaned_pos, cut_threshold, True)
    print("build vocab data neg")
    vocab_neg = build_vocab_counter(cleaned_neg, cut_threshold, True)
    print("build vocab test data")
    vocab_test = build_vocab_counter(cleaned_test, cut_threshold, True)


    print("export data")
    #export tweets
    if(is_full):

        #export tweets
        export(cleaned_pos,  "cleaned_pos_full_bitri=True")
        export(cleaned_neg,  "cleaned_neg_full_bitri=True")
        export(cleaned_test, "cleaned_test_bitri=True")

        #export vocabs
        write_vocab_to_file(vocab_pos, "cleaned_vocab_pos_full_bitri=True")
        write_vocab_to_file(vocab_neg, "cleaned_vocab_neg_full_bitri=True")
        write_vocab_to_file(vocab_test, "cleaned_vocab_test_bitri=True")


    else :
        #export tweets
        export(cleaned_pos,  "cleaned_pos_bitri=True")
        export(cleaned_neg,  "cleaned_neg_bitri=True")
        export(cleaned_test, "cleaned_test_bitri=True")

        #export vocabs
        write_vocab_to_file(vocab_pos, "cleaned_vocab_pos_bitri=True")
        write_vocab_to_file(vocab_neg, "cleaned_vocab_neg_bitri=True")
        write_vocab_to_file(vocab_test, "cleaned_vocab_test_bitri=True")



#----------------------------no_dot(tweet)------------------------------------
#replace dots (but not "...") by a space in a tweet
#tweet : the tweet in which dots should be replaced


def no_dot(tweet):
    tweet = tweet.split()
    for i, t in enumerate(tweet):
        if t == len(t) * ".": #if contains only dots
            if len(t) > 3:
                tweet[i] = "..."
            continue
        if "." in t:
            if t[-1] == ".":
                t = t[:-1]

            t = t.replace("."," ")
            tweet[i] = t
    return " ".join(tweet)



#----------------------------find_repetition(tweet)------------------------------------
#replace occurences of n successive similar letters by a single one, where n >= 3
#tweet : the tweet in which repetitions should be removed


def find_repetition(tweet):
    copy = ""
    i = 0
    while i<len(tweet):
        if (i<len(tweet)-2 and tweet[i] == tweet[i+1] and tweet[i] == tweet[i+2] and tweet[i]!='.'):
            i = i+2
            while( i<len(tweet)-1 and tweet[i] == tweet[i+1] ):
                i+=1
            copy += tweet[i]
            i+=1
        else:
            copy += tweet[i]
            i+=1
    return copy


#----------------------------stem_tweet(tweet)------------------------------------
#stem all the words in a tweet using the nltk stemmer
#tweet : the tweet in which words should be stemmed
'''
def stem_tweet(tweet):
    ps = PorterStemmer()
    tweet = tweet.split()

    #stem each word of the tweet
    for i, t in enumerate(tweet):
        tweet[i] = ps.stem(t)

    #reassemble the tweet words
    return " ".join(tweet)
'''

#----------------------------standardize_tweets(tweets)------------------------------------
#remove repetitions and dots in the tweets
#tweets : the tweets in which repetitions and dots should be avoided, a list of strings

def standardize_tweets(tweets):
    "filter and uniform the tweets "

    ps = PorterStemmer()
    tknzr = TweetTokenizer(preserve_case=False)
    loading_counter = 0
    processed_tweets = []
    for i, tweet in enumerate(tweets):
        tweet = find_repetition(tweet)
        tweet = no_dot(tweet)
        tweet = " ".join([process_word(ps, word) for word in tokenize(tknzr, tweet)])

        processed_tweets.append(tweet)

        if loading_counter%1000==1:
            print("{:.1f}".format(loading_counter/len(tweets)*100), "%", end='\r')
        loading_counter+=1


    return processed_tweets



#----------------------------extract_tokens(tknzr, tweet_string)------------------------------------
#extract unigrams, bigrams and trigrams from a tweet
#tweet_string : the tweet from which n-grams should be extracted
#tknzr : the tokenizer charged to extract the tokens, we use the one from nltk library

def extract_tokens(tknzr, tweet_string):

    unigrams = [token for token in tokenize(tknzr, tweet_string)]

    def generator(unigrams):
        yield from [tuple([u]) for u in unigrams]
        yield from bigrams(unigrams)
        yield from trigrams(unigrams)

    return generator(unigrams)


#----------------------------build_vocab_counter(tweets, cut_threshold, bitri)------------------------------------
#retun a counter of n-grams occurences from a set of tweets
#tweets : the set of tweets from which the counter counts
#cut_threshold : the counter returned knows no wrds that have less than cut_threshold occurences in tweets
#bitri : True iff the counter should count bigrams and trigrams of the words, counts only ords iff False

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
            final_counter.update(tokenize(tknzr, tweet))
        if loading_counter%5000==1:
                print("{:.1f}".format(loading_counter/len(tweets)*100), "%", end='\r')
        loading_counter+=1

    # remove less frequent items
    print("removing items present less than", cut_threshold, "times")
    final_counter = Counter(token for token in final_counter.elements() if final_counter[token] >= cut_threshold)

    timeElapsed=datetime.now()-startTime

    print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

    return final_counter



#----------------------------contains(tweet, words)------------------------------------
#return true iff the tweet contains at least one of the words
#tweet is the thweet we want to know if it contains one of the words
#words is a list of words thant we want to know if one is contained in the tweet

def contains(tweet, words):
    for w1 in tweet.split():
        for w2 in words:
            if w1 == w2:
                return True
    return False

#----------------------------drop_duplicates(tweets)------------------------------------
#eliminate duplicated tweets
#tweets is the list of tweets in which we want no duplicate

def drop_duplicates(tweets):
    tweets = list(set(tweets))
    return tweets


#----------------------------characteristic_words(data_tweets, merged)------------------------------------
#writes in a file the words that can be considered as characteristics (see the report for further information)
#data_tweets : the tweets that need to be labelled
#merged : a dataframe containing pos words and neg words


def characteristic_words(data_tweets, merged, is_full):
    #determine min number of occurence to define a word as characteristic
    min_diff = set_min_diff(data_tweets, merged)

    #create positive characteristic words
    char_pos_words = merged[(merged.ratio == 1) & (merged.occurence_pos >= min_diff)]
    char_pos_words = char_pos_words[["word","occurence_pos"]]

    #save characteistic words to .txt file
    char_pos_words.set_index("word")
    if is_full:
        char_pos_words.to_csv(sep="\t", path_or_buf="characteristc_pos_full_words.txt", header=False, index=False)
    else:
        char_pos_words.to_csv(sep="\t", path_or_buf="characteristc_pos_words.txt", header=False, index=False)

    #create negative characteristic words
    char_neg_words = merged[(merged.ratio == 1) & (merged.occurence_neg >= min_diff)]
    char_neg_words = char_neg_words[["word","occurence_neg"]]

    #save characteistic words to .txt file
    char_neg_words.set_index("word")
    if is_full:
        char_neg_words.to_csv(sep="\t", path_or_buf="characteristic_neg_full_words.txt", header=False, index=False)
    else:
        char_neg_words.to_csv(sep="\t", path_or_buf="characteristic_neg_words.txt", header=False, index=False)


#----------------------------set_min_diff(data_tweets, merged)------------------------------------
#determines what is the minimum number of occurrences for a word to be characteristic
#data_tweets : the tweets that need to be labelled
#merged : a dataframe containing pos words and neg words

#Takes tweets to be labelled, and a merged instance of pos and neg vocabs
def set_min_diff(data_tweets, merged):
    ratio_one = merged[(merged.ratio == 1)]
    differences = (sorted(list(ratio_one.difference), reverse=True))
    previous = -1

    for d in differences :
        found = True

        #do not compute the same difference twice
        if d != previous :
            previous = d

            #keep only words whose number of occurences is > d
            pos_words = ratio_one[ratio_one["occurence_pos"] > d]
            neg_words = ratio_one[ratio_one["occurence_neg"] > d]

            for t in data_tweets:

                #d is not acceptable if two opposite characteristic words can be found in the same test tweet
                if contains(t, pos_words) and contains(t, neg_words):
                    found= False
                    break;

            if(found):
                return d
            else:
                print(str(d) + "not sucessful")
    return max(differences)


#----------------------------process_word(ps, string)------------------------------------
#pre-process a word using the nltk stemmer
#ps : the PorterStemmer necessary to stem the word
#string : the word to be pre-processed


def process_word(ps, string):
    string = ps.stem(string)
    if len(string) > 8:
        string = string[:8]
    if (string.startswith("haha") or string.startswith("ahah")) and not(False in[c=="a" or c=="h" for c in string]):
        string = "haha"
    return string

#----------------------------build_global_vocab(is_full, bitri, cut)------------------------------------
#creates a vocab of words and their occurences, words being contained in either pos, neg ort test cleaned tweets
#is_full : True iff pos, neg and test that should be considered are the full versions
#bitri : True iff we also want to add n-grams to the vocab
#cut : the minimal number of occurences that a word contained in the final vocab should have

def build_global_vocab(is_full, bitri, cut):


    if is_full:
        full_string = "full_"
    else :
        full_string = ""

    #import tweets
    pos = import_("cleaned_pos_"+str(full_string)+"bitri="+str(bitri))
    neg = import_("cleaned_neg_"+str(full_string)+"bitri="+str(bitri))
    test = import_("cleaned_test_bitri="+str(bitri))


    #create counter for all words
    pos_counter = build_vocab_counter(pos, cut, bitri)
    neg_counter = build_vocab_counter(neg, cut, bitri)
    test_counter = build_vocab_counter(test, cut, bitri)

    #sum counts
    global_counter = pos_counter + neg_counter + test_counter

    #write vocab file
    write_vocab_to_file(global_counter, "global_"+str(full_string)+"vocab_cut=" + str(cut))
