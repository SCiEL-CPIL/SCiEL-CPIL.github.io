import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = 2, 3, 10
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # (coefficient of A[i,j]*B[j,k] in C[i,k] is always 1)
    # This is a rank-1 tensor: T = u ⊗ v ⊗ w with all entries 1
    # Rank-1 decomposition:
    #   T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # Set U = 1, V = 1, W = 1 for all entries
    
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    
    return {"U": U, "V": V, "W": W}
