"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the tensor decomposition.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    
    # Known optimal ranks for small matrix multiplication tensors
    # If a known better decomposition exists, use it.
    # For 3x4x7, schoolbook is 84. No standard better decomposition is widely known/implemented 
    # in standard libraries for this specific asymmetric case that is easily verifiable 
    # and guaranteed to be exact with integer/half-integer entries in a short time.
    # We default to the schoolbook decomposition which is exact and valid.
    
    rank = n * m * p
    
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
