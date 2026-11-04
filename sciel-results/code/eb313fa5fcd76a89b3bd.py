import numpy as np
import time

def construct():
    """
    Construct a rank-36 decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy: Direct term-by-term decomposition.
    C[i,k] = sum_j A[i,j] B[j,k].
    There are 4 choices for j and 9 choices for k.
    Total 36 terms.
    
    For each (j, k) pair, we define a rank-1 term.
    U selects A[:, j].
    V selects B[j, k].
    W adds to C[:, k].
    """
    
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    rank = m * p  # 36
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U: Select A[i, j] for all i.
            # Index in U is i*m + j.
            for i in range(n):
                U[i * m + j, r] = 1.0
            
            # V: Select B[j, k].
            # Index in V is j*p + k.
            V[j * p + k, r] = 1.0
            
            # W: Add to C[i, k] for all i.
            # Index in W is k*n + i.
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {"U": U, "V": V, "W": W}
