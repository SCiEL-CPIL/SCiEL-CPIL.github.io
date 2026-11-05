import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    rank = 1
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # All-ones tensor decomposition: T[i,j,k] = 1 = 1*1*1
    U[:, 0] = 1
    V[:, 0] = 1
    W[:, 0] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
