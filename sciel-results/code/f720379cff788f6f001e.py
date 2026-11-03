"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use a rank-50 construction based on grouping the p dimension into pairs
    # and using a bilinear form that exploits the structure of the matrix product.
    
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # The idea: for each i in range(n), j in range(m), and each pair of k's,
    # we can use a single tensor rank term to handle two k's at once if we
    # allow half-integer coefficients. Specifically, we can group the 10 columns
    # into 5 pairs, and for each pair use 10 terms (2 for each of the 2 rows of A? no).
    # Actually, let's use a different approach: we can use the fact that the product
    # C_{i,k} = sum_j A_{i,j} B_{j,k} can be computed by precomputing certain
    # linear combinations.
    
    # Let's use a construction that has rank = n * m * ceil(p/2) = 2 * 3 * 5 = 30? 
    # No, that's not possible. The lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
    # But we need to check if rank 30 is achievable.
    
    # Actually, for matrix multiplication of size (n x m) and (m x p), the tensor rank
    # is at most n*m*p, and at least max(n*m, m*p, p*n). For 2x3 and 3x10, the lower
    # bound is 30. Let's see if we can achieve rank 30.
    
    # We can use the following construction: for each j in range(m) and each k in range(p),
    # we define a term that contributes to C_{i,k} for all i. But that would give rank m*p = 30.
    # Let's check: term (j, k) has U_{i*m+j} = A_{i,j} (but A is a matrix, not a vector).
    # This doesn't work directly.
    
    # Let's use the standard construction: rank = n*m*p = 60.
    # But we can do better. Let's use the following:
    # For each j in range(m), define m_j = 10 (the number of columns in B).
    # We can use a rank of m * max(n, p) = 3 * 10 = 30? No, that's not right.
    
    # Actually, the standard lower bound for the tensor rank of matrix multiplication
    # is max(n*m, m*p, p*n) = 30. But it's not always achievable.
    
    # Let's use a construction that achieves rank 50, which is better than 60.
    # We can group the 10 columns of B into 5 pairs, and for each pair, use 10 terms
    # (2 for each of the 5 pairs? no).
    
    # Actually, let's use the following construction:
    # For each i in range(n), j in range(m), and k in range(p), we have a term.
    # But we can combine terms: for fixed i and j, we can combine all k's into a single
    # term if we use a vector. But that doesn't work.
    
    # Let's just use the naive construction for now, as it's correct.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
