import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    # The tensor for matrix multiplication M_{n,m,p} has entries T_{ijk} = 1.
    # It can be decomposed as a sum of rank-1 tensors corresponding to each j.
    # T_{ijk} = sum_{j'} delta_{j,j'} * 1 * 1 * 1
    # Each term is rank 1.
    # Total rank is m.
    rank = m
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        # U corresponds to A_{ij}. We need U_{i, r, r} = 1 for all i.
        # Index in U: i * m + j. Here j=r.
        for i in range(n):
            U[i * m + r, r] = 1.0
            
        # V corresponds to B_{jk}. We need V_{r, k, r} = 1 for all k.
        # Index in V: j * p + k. Here j=r.
        for k in range(p):
            V[r * p + k, r] = 1.0
            
        # W corresponds to C_{ki}. We need W_{k, i, r} = 1 for all k, i.
        # Index in W: k * n + i.
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
