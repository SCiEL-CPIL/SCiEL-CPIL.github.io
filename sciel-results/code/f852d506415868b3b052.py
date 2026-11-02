import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit

    n, m, p = DIMENSIONS
    
    # First, build the rank-35 Strassen decomposition
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for blk in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(4):
                U[idx][col] = a_vec[idx]
            V[0 * p + 2 * blk][col] = b_vec[0]
            V[0 * p + 2 * blk + 1][col] = b_vec[1]
            V[1 * p + 2 * blk][col] = b_vec[2]
            V[1 * p + 2 * blk + 1][col] = b_vec[3]
            W[2 * blk * n + 0][col] = c_vec[0]
            W[2 * blk * n + 1][col] = c_vec[1]
            W[(2 * blk + 1) * n + 0][col] = c_vec[2]
            W[(2 * blk + 1) * n + 1][col] = c_vec[3]
            col += 1
    
    # Now check for linear dependencies using PSLQ-like approach
    # Each rank-r term contributes a vector in R^{4*20*20}... no.
    # The tensor T has 40 entries. Each rank-1 term u⊗v⊗w gives a 40-vector
    # where entry (i,k,j) = U[i,k,r]*V[k,j,r]*W[j,i,r]
    # We need sum_r of these 40-vectors to equal T (the matrix mult tensor)
    
    # Build the 40-vector for each rank term
    def term_vector(r):
        vec = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    vec.append(U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r])
        return vec
    
    # Build matrix where column r is the 40-vector of term r
    # Check rank of this 40x35 matrix
    M = np.zeros((40, rank))
    for r in range(rank):
        M[:, r] = term_vector(r)
    
    # Use SVD to check numerical rank
    try:
        sv = np.linalg.svd(M, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        num_rank = np.sum(sv > tol)
        if num_rank < rank:
            # There are linear dependencies! We can potentially reduce rank
            # For now, just note this - the Strassen terms should be independent
            pass
    except Exception:
        pass
    
    # The 35 Strassen terms for 2x2x10 are expected to be linearly independent
    # since each block's 7 terms are independent and blocks don't interact
    # Return the rank-35 decomposition
    return {"U": U, "V": V, "W": W}
