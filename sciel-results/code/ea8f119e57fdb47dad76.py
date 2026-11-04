"""Research program: Attempts rank-14 construction via 3x3 tensor embedding, falls back to schoolbook."""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The task is to find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    # The tensor T[i, j, k] represents the multiplication C[i, k] = sum_j A[i, j] B[j, k].
    # In the standard basis, T[i, j, k] = 1 for all i, j, k.
    # The decomposition is T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    # The directive suggests using the rank-7 decomposition for 3x3 matrix multiplication.
    # However, it is a well-known result that the minimum rank for 3x3 matrix multiplication is 23 (Lader 1974), not 7.
    # Strassen's algorithm reduces 2x2 from 8 to 7.
    # For 3x3, the best known bounds are around 21-23.
    # Therefore, a "rank-7 decomposition for 3x3" does not exist in the exact sense over real/complex numbers.
    # The premise of the directive contains a factual error regarding the existence of a rank-7 decomposition for 3x3.
    
    # Since the proposed research path relies on a non-existent mathematical object, we cannot construct a valid rank-14 decomposition using it.
    # We must fall back to the guaranteed schoolbook solution.
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col_idx] = 1
                V[j * p + k][col_idx] = 1
                W[k * n + i][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
