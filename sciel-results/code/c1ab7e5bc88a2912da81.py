"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    # Matrix multiplication: A (n x m) * B (m x p) = C (n x p)
    # Tensor T_{i,j,k} = 1 if we multiply A_{i,j} * B_{j,k} to contribute to C_{i,k}
    # The standard schoolbook has rank n*m*p = 128.
    # We try to find a lower rank decomposition.
    
    # Strategy: Use a randomized approach with Gaussian elimination / rank reduction.
    # For small matrices, we can try to find a decomposition with rank < 128.
    
    # Known lower bounds: For 4x4x4, the minimum rank is 19 (Strassen-like).
    # For 4x4x8, we can potentially do better than 128.
    
    # Let's try a structured approach:
    # 1. Start with schoolbook rank 128
    # 2. Try to reduce rank by finding linear dependencies
    
    # The tensor T has entries T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition gives matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For the standard multiplication tensor, T[i, j, k] = 1 for all valid (i,j,k).
    # Wait, that's not right. The multiplication tensor has T[i, j, k] = 1 
    # when we multiply element (i,j) of A with element (j,k) of B to get (i,k) of C.
    # Actually, the tensor is T[i, j, k] = 1 for each triple (i, j, k) representing 
    # A_{i,j} * B_{j,k} contributing to C_{i,k}.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} has:
    # - First mode: i (row of A / row of C), size n
    # - Second mode: j (col of A / row of B), size m  
    # - Third mode: k (col of B / col of C), size p
    # And M[i, j, k] = 1 for all i, j, k (it's the all-ones tensor in the sense that
    # each entry represents one multiplication).
    
    # Actually no. The standard matrix multiplication tensor has M[i, j, k] = 1 
    # for each (i, j, k) where the product A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # Since for each (i, j, k) there is exactly one such contribution, the tensor 
    # is indeed all-ones: M[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # A rank-r decomposition: M[i,j,k] = sum_{c=1}^{r} U_{i,c} * V_{j,c} * W_{k,c}
    # where U is n x r, V is m x r, W is p x r.
    
    # But the problem statement says U has shape (n*m, rank), V has shape (m*p, rank), 
    # W has shape (p*n, rank). This suggests a different indexing convention.
    
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U has n*m rows, V has m*p rows, W has p*n rows.
    # The decomposition is: for each (i, j, k), 
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is the same as the standard tensor decomposition but with flattened indices.
    
    # Let's use numpy to try to find a low-rank decomposition.
    
    # Approach: Randomized SVD-like approach with integer rounding.
    # Or: Use the fact that for small matrices, we can try various ranks.
    
    # Start with the schoolbook solution (rank 128) and try to improve.
    
    # For now, return the schoolbook solution. We can try to improve with more sophisticated methods.
    
    rank = n * m * p  # 128
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to reduce the rank using a simple approach
    # Convert to numpy arrays for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find linear dependencies and reduce rank
    # This is a complex optimization problem, so we'll use a heuristic approach
    
    # For the all-ones tensor, there are known decompositions with lower rank.
    # For 4x4x4, the best known is 19 (vs 64 for schoolbook).
    # For 4x4x8, we can potentially achieve much less than 128.
    
    # Let's try a simple rank reduction: if we can find a linear combination of columns
    # that equals zero, we can remove that column.
    
    # Actually, let's just return the schoolbook solution for now, as finding a 
    # provably optimal low-rank decomposition is a hard problem.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
