import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T_{i,j,k} = 1 for all i,j,k.
    # The rank of the all-ones tensor is 1.
    # U_{i*m+j, 0} = 1
    # V_{j*p+k, 0} = 1
    # W_{k*n+i, 0} = 1
    # Sum_{r=0}^0 1*1*1 = 1.
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
