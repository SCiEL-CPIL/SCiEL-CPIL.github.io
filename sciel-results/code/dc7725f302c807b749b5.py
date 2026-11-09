"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook (naive) decomposition for 4x5x6 matrix multiplication.
    
    The matrix multiplication tensor for C[i,k] = sum_j A[i,j] * B[j,k]
    has entries T[i,j,k] = 1 for all valid i,j,k.
    
    The schoolbook decomposition gives rank = n*m*p = 4*5*6 = 120.
    
    U shape: (n*m, rank) = (20, 120)
    V shape: (m*p, rank) = (30, 120)
    W shape: (p*n, rank) = (24, 120)
    
    For each term c = i*m + j*p + k (or any bijection from (i,j,k) to [0,rank)):
      U[i*m + j, c] = 1
      V[j*p + k, c] = 1
      W[k*n + i, c] = 1
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 120
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Safety margin check - if we're running out of time, return what we have
    # (though for schoolbook this is instantaneous)
    if time.time() - start_time > 140:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
