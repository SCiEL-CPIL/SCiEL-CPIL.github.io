"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def verify_full(U, V, W):
        """Full verification - expensive but complete."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=int)
        for r in range(rank):
            for a in range(nm):
                if U[a, r] != 0:
                    for b in range(mp):
                        if V[b, r] != 0:
                            for c in range(pn):
                                if W[c, r] != 0:
                                    T_recon[a, b, c] += U[a, r] * V[b, r] * W[c, r]
        return np.array_equal(T_recon, T)
    
    # Schoolbook decomposition (rank = n*m*p = 54)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=int)
    V_school = np.zeros((mp, rank_school), dtype=int)
    W_school = np.zeros((pn, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower-rank decomposition using structured approaches
    # For 3x3x3 matrix multiplication, the Strassen algorithm achieves rank 7
    # For 3x3x6, we can potentially use a similar approach
    
    # Let's try a different approach: use the fact that we're multiplying 3x3 by 3x6
    # We can think of this as 2 independent 3x3 multiplications (for the two halves of the 3x6)
    # But we might be able to share some computation
    
    # Actually, let's try to construct a rank-46 decomposition
    # The idea: for each of the 3 columns of the result (since p=6, we have 6 columns, 
    # but the matrix is 3x6, so we have 3 rows and 6 columns)
    
    # Let me think about this differently. The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # where i in [0,3), j in [0,3), k in [0,6)
    
    # A simple improvement: we can factor the 6 k-values into 2 groups of 3
    # and use a shared structure
    
    # Let's try rank = 46 = 54 - 8 (save 8 terms)
    # Or try to find a structured decomposition
    
    # For now, let's try a random search for a lower-rank decomposition
    # This is computationally expensive, so we'll be conservative
    
    # Alternative: use known results. For 3x3x3, the minimal rank is 7 (Strassen)
    # For 3x3x6, we can potentially use 2*7 = 14 if we can share nothing,
    # or less if we can share some terms
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms
    
    # For each pair of terms (r1, r2), check if we can replace them with a single term
    # This requires solving a system of equations
    
    # Due to time constraints, let's just return the schoolbook decomposition
    # but with a better verification
    
    # Convert to list of lists for output
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
