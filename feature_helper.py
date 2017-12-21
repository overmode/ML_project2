import numpy as np
import pickle
from ProcessTweets import *
from nltk.tokenize import TweetTokenizer


def construct_features(vocab_filename, tweets_filename, embeddings_filename, relevant_filename, nb_concat):
    
    vocab = extract_index(vocab_filename)
    tweets = import_(tweets_filename)
    embeddings = np.load(embeddings_filename)
    relevant = extract_relevant(relevant_filename)
    features = []
    tknzr = TweetTokenizer(False)
    nb_dim = 50
    if nb_concat == -1:
        nb_concat = len(tweets)
    
    tweets_embeddings = []
    
    #get the embeddings of each token
    loading_counter = 0
    for tweet in tweets:
        token_embeddings = []
        for token in extract_tokens(tknzr, tweet):
            if token in vocab:
                index = vocab.get(token)
                token_embeddings.append([embeddings[index], float(relevant.get(token, 0))])
                
        #sum the different embeddings
        if len(token_embeddings)==0:
            tweets_embeddings.append([0]*nb_dim)
            continue
        sorted_token_embeddings = sorted(token_embeddings, key=lambda x: x[1])
        sum_token_embeddings = sorted_token_embeddings[0][0]
        for token_embedding in sorted_token_embeddings[1:nb_concat]:
            sum_token_embeddings = sum_token_embeddings + token_embedding[0]
            
        tweets_embeddings.append(sum_token_embeddings / nb_concat)
        
        if loading_counter%1000==1:
            print("{:.1f}".format(loading_counter/len(tweets)*100), "%", end='\r')
        loading_counter+=1
            
    return tweets_embeddings


def policy_unpredictable():
    return np.random.choice((1,-1))

def assemble(valid, indices):
    cur = 0
    nb_inserted = 0
    result = [0]*(len(valid) + len(indices))
    for i in range((len(valid) + len(indices))):
        if(cur in indices):
            result[cur] = policy_unpredictable()
            cur = cur + 1
        else:
            result[cur] = valid[nb_inserted]
            cur = cur + 1
            nb_inserted = nb_inserted + 1
    return np.array(result)
