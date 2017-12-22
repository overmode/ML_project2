import pandas as pd  # Pandas will be our framework for the first part
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import TweetTokenizer
from IOTweets import *
from ProcessTweets import *
import numpy as np
from split_data import split_data
from scipy.sparse import *
from sklearn import linear_model, preprocessing, neural_network
import pickle
import random
from feature_helper import *
from pickle_vocab import *
from cooc import *
from subprocess import call




#----------------------------------------------setup(path_pos, path_neg, path_test, is_full,  pertinence_thres_relevant, min_count_relevant, cut_threshold=5, bitri = True)-----------------------------------

#setup corresponds to the cleaning phase indicated in the report.
#path_pos : the path of the file containing full raw positive tweets
#pathj neg : the path of the file containing full raw negative tweets
#path_test : the path of the file containing full raw test tweets
#is_full : True iff we consider the full version of tweets (thus True for kaggle submission) 
#pertinence_thres_relevant : a relevant word should not have a pertinence smaller than pertinence_thres_relevant
#min_count_relevant : a relevant word should not have a total occurences number smaller than min_count_relevant
#cut_threshold : a word in a vocab should not have a global number of occurences smaller than cut_threshold
#bitri : True iff we want the vocabs to consider n-grams as well


#clean raw tweets
def setup(path_pos, path_neg, path_test, is_full,  pertinence_thres_relevant, min_count_relevant, cut_threshold=5, bitri = True):
    #path pos, path_neg, path_data : paths of the raw tweets
    #is_full : True iff we clean 
    
    #export cleaned tweets and their vocabulary
    cleaned_tweets = clean_tweets(path_pos, path_neg, path_test, is_full, cut_threshold)
    
    if is_full:
        full_string = "_full"
    else :
        full_string = ""
    
    
    #load dataframes    
    pos_df = build_df("cleaned_vocab_pos"+str(full_string)+"_bitri="+str(bitri), bitri)
    neg_df = build_df("cleaned_vocab_neg"+str(full_string)+"_bitri="+str(bitri), bitri)
   
    #Merge dataframes
    merged = merging(neg_df, pos_df, False)
    
    #create relevant vocab
    create_relevant_vocab(pertinence_thres_relevant, min_count_relevant, merged)
    
    data_tweets = import_("cleaned_test_bitri="+str(bitri))
    
    #create characteristic_words
    characteristic_words(data_tweets, merged, is_full)
    
    #build the global vocabulary
    build_global_vocab(is_full, bitri, cut_threshold)
    
    
    
def submit(pertinence):
    #Load words from tweet set
    #xs = np.load(embeddings_ts_full)
    
    #define relevant_vocab file to use
    relevant_vocab = 'relevant_vocab_full_lb=1000.txt'
    
    #load ratios into a dictionary
    weights = extract_relevant(relevant_vocab)
    
    print(len(new_vocab))
    
    pos_tweets = np.array(open(file=pos_ts_full_tweets, mode='r', encoding="utf8").readlines()) 
    neg_tweets = np.array(open(file=neg_ts_full_tweets, mode='r', encoding="utf8").readlines()) 
    te_tweets = np.array(open(file=te_full_tweets, mode='r', encoding="utf8").readlines()) 

    
    #Find features for each tweet with at least one word within vocab, get indices of unpredictable tweets
    #for training tweets
    pos_ts_full_feat, invalid_pos_ts_full = construct_features("global_vocab_cut=5", pos_tweets, "embeddings_bitri=True.npy", "relevant_vocab_pert=0.3_count=300", 5)

    neg_ts_full_feat, invalid_neg_ts_full = construct_features("global_vocab_cut=5", neg_tweets, "embeddings_bitri=True.npy", "relevant_vocab_pert=0.3_count=300", 5)
    
    print(len(pos_ts_full_feat), len(invalid_pos_ts_full) )
    print(len(neg_ts_full_feat), len(invalid_neg_ts_full) )
    
    #for test tweets
    te_full_feat, invalid_te_full = construct_features("global_vocab_cut=5", te_tweets, "embeddings_bitri=True.npy", "relevant_vocab_pert=0.3_count=300", 5)
    print(len(te_full_feat), len(invalid_te_full) )
    
     #Initialize classifier and scaler
    neural = neural_network.MLPClassifier(hidden_layer_sizes=(nb_dim, nb_dim, nb_dim/2, 2))
    scaler = preprocessing.StandardScaler()

    #fit classifier on predictable tweets
    X = np.concatenate((pos_ts_full_feat, neg_ts_full_feat))
    Y = np.concatenate((np.ones(len(pos_ts_full_feat)), np.full(len(neg_ts_full_feat), -1)))
    X = scaler.fit_transform(X, Y)
    neural = neural.fit(X, Y)

    #scale data that should be predicted
    te_full_feat_scaled = te_full_feat
    te_full_feat_scaled = scaler.fit_transform(te_full_feat_scaled, np.ones(len(te_full_feat))) 


    #predict predictable tweets
    te_prediction = neural.predict(te_full_feat_scaled)

    #merge with unpredictable tweets predictions
    labels = assemble(te_prediction, invalid_te_full)
   
    return labels
    
    
    
def GloVe(file_name="cooc_partial", destination='embeddings_ts.npy', embedding_dim = 50):
    #load coocurence matrix
    with open(file_name, 'rb') as f:
        cooc = pickle.load(f)    
    
    nmax = 100
    
    eta = 0.001
    alpha = 3 / 4
    epochs = 10
    
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))
   
    #Construct vector representations xs for words
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        loading_counter = 0
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            
            f = ((n / nmax)**alpha) if n < nmax else 1
            inter_cost = (xs[ix]@(ys[jy]) - np.log(n))
            # We compute the gradients for both context and main vector words
            grad_main = f * inter_cost * ys[jy]
            grad_context = f * inter_cost * xs[ix]
    
            # Update the vector words
            xs[ix] = xs[ix] - (eta * grad_main)
            ys[jy] = ys[jy] - (eta * grad_context)
            
            if loading_counter%20000==1:
                    print("{:.1f}".format(loading_counter/len(cooc.col)*100), "%", end='\r')
            loading_counter+=1
            
    #Store xs in destination file
    np.save(file=destination, arr=xs)
    print("finished")
    
    
    
    
    
    
    
    
    
    
    
#build the cleaned tweets, relevant vocab and characteristic words
setup("train_pos_full.txt", "train_neg_full.txt" , "test_data.txt", is_full=True, pertinence_thres_relevant=0.3, min_count_relevant=250)

# Build the global vocab, which contains: pos, neg and the test vocab
build_global_vocab(False, True, 5)

# Create the vocab from the global vocab
to_vocab_pickle("global_vocab_cut=5")

# Create the cooc matrix from the global vocab
to_cooc(path_vocab="vocab_full.pkl", path_tweets_pos = "cleaned_pos_full_bitri=True", path_tweets_neg = "cleaned_neg_full_bitri=True",path_tweets_test ="cleaned_test_bitri=True", dest = 'cooc_full.pkl')
        
#commande shell to concatenate all tweets
call(["cat","cleaned_pos_full_bitri=True","cleaned_neg_full_bitri=True","cleaned_test_bitri=True"], stdout=open("all_tweets_cleaned", 'w'));

#Build embeddings using Glove
GloVe(file_name="cooc_full.pkl", destination="embeddings_bitri=True")

submit(0.3)




