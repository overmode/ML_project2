import pandas as pd
from datetime import datetime
from collections import Counter
from nltk.stem import PorterStemmer
from nltk import bigrams, trigrams
from nltk.tokenize import TweetTokenizer

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
    relevant = relevant[["word","ratio"]]
    relevant.set_index("word")
    relevant.to_csv(sep="\t", path_or_buf=("relevant_vocab_pert="+str(pertinence_thres)+"_count="+str(min_count)), header=False, index=False)


#----------------------------stemming(is_full, cut_threshold)------------------------------------
#stem the words in tweets and save the updated tweets in a fallen
#is_full : True iff we load the full version of tweets
#cut_threshold : the minimum number of total occurences that words in the vocab of stemmed tweets should have

def stemming(is_full, cut_threshold):

    print("Import Data")

    if(is_full):
        #import tweets
        preprocessed_neg = import_("preprocessed_neg_full")
        preprocessed_pos = import_("preprocessed_pos_full")
        preprocessed_test = import_("preprocessed_test_full")

        #import vocabs
        neg_df = build_df("preprocessed_vocab_neg_full")
        pos_df = build_df("preprocessed_vocab_pos_full")
        test_df = build_df("preprocessed_vocab_test_full")

    else :
        #import tweets
        preprocessed_neg = import_("preprocessed_neg")
        preprocessed_pos = import_("preprocessed_pos")
        preprocessed_test = import_("preprocessed_test")

        #import vocabs
        neg_df = build_df("preprocessed_vocab_neg")
        pos_df = build_df("preprocessed_vocab_pos")
        test_df = build_df("preprocessed_vocab_test")



    #Merge Pos and neg dataframes
    merged = merging(neg_df, pos_df, True)
    merged = merging(merged, test_df, True)


    print("Building semantic")

    #Build equivalence list
    same_begin = list(merged[merged["len"]==8]["word"])
    same_begin = [list(merged.loc[(merged.word.str.startswith(w))]["word"]) for w in same_begin]

    print("Building haha semantic")

    #build "haha" equivalences
    word_haha_list = list(merged.loc[merged.word.str.startswith("haha") | merged.word.str.startswith("ahah")]["word"])

    word_haha_list.remove("haha")
    word_haha_list.insert(0, "haha")
    same_begin.append(list(word_haha_list))



    print("filter semantic")

    #fiter roots alone
    semantic = filter_single_rep(same_begin)


   #process tweets
    print("Process data pos")
    stemmed_pos = stem_tweets( preprocessed_pos, semantic)
    print("Process data neg")
    stemmed_neg = stem_tweets( preprocessed_neg, semantic) #Sous form d'un tableau de tweet
    print("Process data test")
    stemmed_test = stem_tweets( preprocessed_test, semantic) #Sous form d'un tableau de tweet



    print("export data")

    #export data
    if(is_full):
        #export tweets
        export(stemmed_neg,  "cleaned_neg_full")
        export(stemmed_pos,  "cleaned_pos_full")
        export(stemmed_test, "cleaned_test_full")

        #export vocabs

        vocab_neg_full = build_vocab_counter(stemmed_neg, cut_threshold,  True)
        vocab_pos_full = build_vocab_counter(stemmed_pos, cut_threshold,  True)
        vocab_test_full = build_vocab_counter(stemmed_test, cut_threshold,  True)

        write_vocab_to_file(vocab_pos_full, "cleaned_vocab_pos_full")
        write_vocab_to_file(vocab_neg_full, "cleaned_vocab_neg_full")
        write_vocab_to_file(vocab_test_full, "cleaned_vocab_test_full")

    else :
        #export tweets
        export(stemmed_neg,  "cleaned_neg")
        export(stemmed_pos,  "cleaned_pos")
        export(stemmed_test, "cleaned_test")

        #export vocabs
        vocab_neg = build_vocab_counter(stemmed_neg, cut_threshold,  True)
        vocab_pos = build_vocab_counter(stemmed_pos, cut_threshold,  True)
        vocab_test = build_vocab_counter(stemmed_test, cut_threshold,  True)

        write_vocab_to_file(vocab_pos, "cleaned_vocab_pos")
        write_vocab_to_file(vocab_neg, "cleaned_vocab_neg")
        write_vocab_to_file(vocab_test, "cleaned_vocab_test")


#----------------------------process_data_no_stem(path_pos, path_neg, path_test, is_full, cut_threshold)------------------------------------
#remove dots and repetitions in tweets, export the preprocessed tweets and their associated vocabs
#path_pos : path towards the file containing pos tweets
#path_neg : path towards the file containing neg tweets
#path_test : path towards the file containing test tweets
#is_full : True iff we load the full version of tweets
#cut_threshold : min number of occurence that a word contained in the vocab should have


def process_data_no_stem(path_pos, path_neg, path_test, is_full, cut_threshold):

    print("Import Data")
    #import tweets
    tweets_test = import_without_id(path_test)
    tweets_pos =  import_(path_pos)
    tweets_neg =  import_(path_neg)

    #remove duplicates
    tweets_pos = drop_duplicates(tweets_pos)
    tweets_neg = drop_duplicates(tweets_neg)

    #process tweets
    print("Process data pos")
    preprocessed_pos = standardize_tweets( tweets_pos)
    print("Process data neg")
    preprocessed_neg = standardize_tweets( tweets_neg) #Sous form d'un tableau de tweet
    print("Process data test")
    preprocessed_test = standardize_tweets( tweets_test) #Sous form d'un tableau de tweet

    #build the counters
    print("build vocab data pos")
    vocab_pos = build_vocab_counter(preprocessed_pos, cut_threshold, False)
    print("build vocab data neg")
    vocab_neg = build_vocab_counter(preprocessed_neg, cut_threshold, False)
    print("build vocab test data")
    vocab_test = build_vocab_counter(preprocessed_test, cut_threshold, False)


    print("export data")
    #export tweets
    if(is_full):

        #export tweets
        export(preprocessed_pos,  "preprocessed_pos_full")
        export(preprocessed_neg,  "preprocessed_neg_full")
        export(preprocessed_test, "preprocessed_test_full")

        #export vocabs
        write_vocab_to_file(vocab_pos, "preprocessed_vocab_pos_full")
        write_vocab_to_file(vocab_neg, "preprocessed_vocab_neg_full")
        write_vocab_to_file(vocab_test, "preprocessed_vocab_test_full")


    else :
        #export tweets
        export(preprocessed_pos,  "preprocessed_pos")
        export(preprocessed_neg,  "preprocessed_neg")
        export(preprocessed_test, "preprocessed_test")

        #export vocabs
        write_vocab_to_file(vocab_pos, "preprocessed_vocab_pos")
        write_vocab_to_file(vocab_neg, "preprocessed_vocab_neg")
        write_vocab_to_file(vocab_test, "preprocessed_vocab_test")


#----------------------------sem_by_repr(semantics, representative, tweet)------------------------------------
#replace all the words in semantic by the representative in a given tweets
#semantics : words that should be mapped on the representative
#representative : the root on which all words in semantic should be mapped
#tweet : the tweet in which words should be mapped

def sem_by_repr(semantics, representative, tweet):
    """Retrun a tweet that countain only the representative of a given semantic """

    for semantic in semantics:
        if (semantic in tweet):
            tweet = tweet.replace(semantic,representative)
    return tweet


#----------------------------sem_by_repr2(semantics, tweet)------------------------------------
#map words in a tweet using a list of semantics
#semantics : a list of list of words where the first word in each inner list is the representative.
#tweet : the tweet in which words should be mapped

def sem_by_repr2(semantics, tweet):
    """Semanticss is a list of list of semantic where the first one is the representative.
    Retrun a tweet that countain only the representative of a given semantic below is an exemple of the semantics

    [['because', 'becausee'],
    ['someone', "someone's", 'someones', 'someonee'],
    ['friends', 'friendship', 'friendships', 'friendss', 'friendster']]
    """

    # We sort by lenth on a non Ascending way to avoid the case where a string is a substring and would chang
    for semantic in semantics:
        representative = semantic[0]
        sem = semantic[1:]
        sem = sorted(sem, key =len, reverse= True)


        tweet = sem_by_repr(sem , representative, tweet)
    return tweet

#----------------------------no_dot(tweet)------------------------------------
#replace dots (but not "...") by a space in a tweet
#tweet : the tweet in which dots should be replaced


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



#----------------------------find_repetition(tweet)------------------------------------
#replace occurences of n successive similar letters by a single one, where n >= 3
#tweet : the tweet in which repetitions should be removed


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


#----------------------------stem_tweet(tweet)------------------------------------
#stem all the words in a tweet using the nltk stemmer
#tweet : the tweet in which words should be stemmed

def stem_tweet(tweet):
    ps = PorterStemmer()
    tweet = tweet.split()

    #stem each word of the tweet
    for i, t in enumerate(tweet):
        tweet[i] = ps.stem(t)

    #reassemble the tweet words
    return " ".join(tweet)


#----------------------------standardize_tweets(tweets)------------------------------------
#remove repetitions and dots in the tweets
#tweets : the tweets in which repetitions and dots should be avoided, a list of strings

def standardize_tweets(tweets):
    "filter and uniform the tweets "

    for i, tweet in enumerate(tweets):
        tweet = find_repetition(tweet)
        tweet = no_dot(tweet)
        #tweets[i] = no_s(tweet)

    return tweets


#----------------------------stem_tweets(tweets, semantic)------------------------------------
#stem the tweets
#tweets : the tweets that should be stemmed, a list of strings

def stem_tweets(tweets, semantic):
    "filter and uniform the tweets "

    for i, tweet in enumerate(tweets):
        tweet = sem_by_repr2(semantic, tweet)
        tweets[i] = stem_tweet(tweet)

    return tweets

#----------------------------extract_tokens(tknzr, tweet_string)------------------------------------
#extract unigrams, bigrams and trigrams from a tweet
#tweet_string : the tweet from which n-grams should be extracted
#tknzr : the tokenizer charged to extract the tokens, we use the one from nltk library

def extract_tokens(tknzr, tweet_string):

    unigrams = [token for token in tknzr.tokenize(tweet_string)]

    def generator(unigrams):
        yield from unigrams
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
            final_counter.update(tweet.split(" "))
        if loading_counter%1000==1:
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


def characteristic_words(data_tweets, merged):
    #determine min number of occurence to define a word as characteristic
    min_diff = set_min_diff(data_tweets, merged)

    #create positive characteristic words
    char_pos_words = merged[(merged.ratio == 1) & (merged.occurence_pos >= min_diff)]
    char_pos_words = char_pos_words[["word","occurence_pos"]]

    #save characteistic words to .txt file
    char_pos_words.set_index("word")
    char_pos_words.to_csv(sep="\t", path_or_buf="characteristc_pos_words.txt", header=False, index=False)

    #create negative characteristic words
    char_neg_words = merged[(merged.ratio == 1) & (merged.occurence_neg >= min_diff)]
    char_neg_words = char_neg_words[["word","occurence_neg"]]

    #save characteistic words to .txt file
    char_neg_words.set_index("word")
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
