"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find a lower rank decomposition using structured approaches
    # Key idea: exploit the structure of 8 = 2*4 or 8 = 2*2*2
    
    # Approach: Try to use Strassen-like tricks on sub-blocks
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # We can try to embed this into the 3x4x8 problem
    
    # Let's try a block decomposition where we split the p-dimension
    # p = 8 can be split into two groups of 4, or four groups of 2
    
    # For each pair of 2x2 sub-blocks in the p-dimension, we could potentially
    # use Strassen's algorithm to save one multiplication
    
    # However, the full 3x4x8 tensor doesn't decompose neatly into independent
    # 2x2x2 blocks because the i and j indices couple across all k values
    
    # Let's try a different approach: random search for linear combinations
    # that can replace multiple schoolbook terms
    
    # Given the complexity, let's try a few specific structured decompositions
    # and verify them
    
    # For now, return the schoolbook decomposition as it's guaranteed correct
    # Future work: implement proper Strassen embedding or other low-rank methods
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
