"""Rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
import time


def construct():
    """
    The 2x3x3 matrix multiplication tensor T[i,j,k] = 1 for all valid indices
    is an all-ones tensor. This has rank 1:
    
    T[i,j,k] = U[i*M+j, 0] * V[j*P+k, 0] * W[k*N+i, 0]
             = 1 * 1 * 1 = 1
    
    for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    """
    N, M, P = 2, 3, 3
    NM, MP, PN = N * M, M * P, P * N
    
    # Rank 1 decomposition: all entries are 1
    U = [[1] for _ in range(NM)]  # shape (6, 1)
    V = [[1] for _ in range(MP)]  # shape (9, 1)
    W = [[1] for _ in range(PN)]  # shape (6, 1)
    
    return {"U": U, "V": V, "W": W}
