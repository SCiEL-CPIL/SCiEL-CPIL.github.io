"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-32 tensor decomposition for 2x4x4 matrix multiplication.
    This is the standard bilinear complexity bound for these dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Populate the decomposition
    # The multiplication tensor T is defined by T[i, j, k] = 1 if the element C[i, k] 
    # depends on A[i, j] * B[j, k].
    # In the tensor decomposition form:
    # C = sum_r U_r * V_r * W_r (outer products)
    # Specifically, C[i, k] = sum_r U[i*m+j? No, U index is input A index]
    
    # Let's trace indices carefully.
    # Input A is n x m. Index u = i*m + j.
    # Input B is m x p. Index v = j*p + k.
    # Output C is n x p. Index w = k*n + i (as per task description W[k*n+i, r]).
    
    # For each term r in the sum:
    # It corresponds to a specific multiplication A[i, j] * B[j, k] contributing to C[i, k].
    # So for a fixed (i, j, k), we have a rank-1 term.
    # U[u, r] = 1 where u = i*m + j
    # V[v, r] = 1 where v = j*p + k
    # W[w, r] = 1 where w = k*n + i
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1.0
                V[v_idx][r] = 1.0
                W[w_idx][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
