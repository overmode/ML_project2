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

        
#commande shell to concatenate all tweets
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
    
print("JSADKJSADKASD")
#Build embeddings using Glove
GloVe(file_name="cooc_full.pkl", destination="embeddings_bitri=True")

submit(0.3)