#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

	for epoch in range(epochs):
		print("epoch {}".format(epoch))
		for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):

            f = ((n / nmax)**alpha) if n < nmax else 1
            inter_cost = (xs[ix]@(ys[iy]) - log(n))

            # We compute the gradients for both context and main vector words
            grad_main = f * inter_cost * ys[iy]
            grad_context = f * inter_cost * xs[ix]

            # Update the vector words
            xs[ix] = xs[ix] - (eta * grad_main)
            ys[iy] = ys[iy] - (eta * grad_context)




    np.save('embeddings', xs)


if __name__ == '__main__':
    main()
