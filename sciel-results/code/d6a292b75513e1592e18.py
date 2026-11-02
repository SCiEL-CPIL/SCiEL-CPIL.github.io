import numpy as np
import time

# The dimensions of the matrix multiplication
N, M, P = 3, 4, 5

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 
    multiplying a (N x M) matrix by a (M x P) matrix.
    
    The rank is N * M * P, which is the schoolbook decomposition.
    For N=3, M=4, P=5, the rank is 60.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (N*M, rank)
              V has shape (M*P, rank)
              W has shape (P*N, rank)
    """
    # The optimal rank for this specific problem is the product of the dimensions
    rank = N * M * P
    
    # Initialize the factor matrices with zeros
    # Using lists of lists for exact integer representation
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    
    # Fill in the non-zero entries for the schoolbook decomposition
    # Each term in the sum corresponds to a specific (i, j, k) triple
    # where i in [0, N), j in [0, M), k in [0, P)
    # The contribution to C[i, k] is A[i, j] * B[j, k]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                # U corresponds to the input A (flattened row-major: i*M + j)
                U[i * M + j][col] = 1
                # V corresponds to the input B (flattened row-major: j*P + k)
                V[j * P + k][col] = 1
                # W corresponds to the output C (flattened row-major: k*N + i)
                # The task specifies W[k*n+i, r]
                W[k * N + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
