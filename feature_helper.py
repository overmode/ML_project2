import numpy as np
import pickle


def construct_features(tweets_filename, embeddings, weights):
    features = []
    invalid_features = [];
    nb_dim = 20
    pertinence = 35

    with open('vocab_full.pkl', 'rb') as f :
        vocab = pickle.load(f)

    print(len(weights))
    #Load words from tweet set
    xs = np.load(embeddings)
    tweets = np.array(open(file=tweets_filename, mode='r').readlines())

    for indl, line in enumerate(tweets):
        #we differentiate tweets containing pertinent words, those in dictionnary 'weights'
        sum_w_pertinent = np.zeros(nb_dim)
        sum_w_others = np.zeros(nb_dim)

        count_pertinent = 0
        count_other = 0

        for word in line.split():
            local_w = vocab.get(word, -1)
            if local_w != -1:
                weight = weights.get(word, -1)
                if weight != -1:

                    #If the word is pertinent, we add its word representation to others pertinents word's representation
                    count_pertinent += weight*pertinence
                    sum_w_pertinent += xs[local_w] * (weight*pertinence)

                else:

                    #If the word is not pertinent, we add its representation to non-pertinent words representations
                    count_other += 1
                    sum_w_others += xs[local_w]

            # If we found pertinent words, we only use them
        if(count_pertinent != 0):
            features.append(sum_w_pertinent/count_pertinent)

            #if we found only non-pertinent words, we use them anyway
        elif count_other!= 0:
            features.append(sum_w_others/count_other)

            #if we did not find words that have representation, we do not try to create features and signal their indices
        else:
            invalid_features.append(indl)

    invalid_features = np.array(invalid_features)
    features = np.array(features)

    return features, invalid_features


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
