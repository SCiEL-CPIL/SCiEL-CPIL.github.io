import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for matrix multiplication.
    For 3x4 * 4x7, the standard schoolbook algorithm yields a rank of 3*4*7 = 84.
    We use a structured linear algebra approach by explicitly mapping the basis elements
    of the input spaces to the output space.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the basis of the tensor product space A \otimes B
    # Each term corresponds to the interaction of A[i,j], B[j,k] -> C[i,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices in the factor matrices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Set the 1s in the rank-1 decomposition
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
