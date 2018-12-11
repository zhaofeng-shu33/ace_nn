import numpy as np
from ace_nn import ace_nn
SIZE = 1000
for rho in [0.1,0.5,0.9]:
    cov_matrix = np.array([[1,rho],[rho,1]]) 
    points = np.random.multivariate_normal([0,0],cov_matrix,SIZE)
    x = points[:,0]
    y = points[:,1]
    h_score = ace_nn(x, y, return_hscore=True)
    print(h_score)