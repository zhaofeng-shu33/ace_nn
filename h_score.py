#!/usr/bin/python
#author: zhaofeng-shu33
# this file uses the rountine ace_nn to estimate {1 \over 2}\norm{\widetilde{B}}_F^2
import numpy as np
from ace_nn import ace_nn

def pearson_correlation(X,Y):
    return (np.mean(X*Y, axis=0) -np.mean(X, axis = 0)* np.mean(Y, axis = 0)) / ( np.std(X, axis = 0) * np.std(Y, axis = 0))

if __name__ == '__main__':
    N_SIZE = 1000
    P_CROSSOVER = 0.7
    x = np.random.choice([0,1],size=N_SIZE)
    n = np.random.choice([0,1], size = N_SIZE, p = [1 - P_CROSSOVER, P_CROSSOVER])
    y = np.mod(x+n, 2)

    # theoretical result = (2 * max(p, 1-p) - 1)^2/2
    # use fortran ace by 1985 article author
    h_score_theoretical = pow(2 * max(P_CROSSOVER, 1 - P_CROSSOVER) - 1, 2)/2
    h_score = ace_nn(x, y, cat = [-1,0], epochs=300, ns=2, return_hscore=True)
    print('h_score_theoretical, h_score', h_score_theoretical, h_score)
    # as can be seen, the transformation is simply a normalization trick
    # tx = \frac{x-E[X]}{\sqrt{\Var[X]}}

