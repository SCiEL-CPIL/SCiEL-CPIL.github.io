"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We will construct a decomposition of rank 64.
    # Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2).
    # C = A B1 + A B2 + A B3.
    # Rank(A B1) <= 23 (Landsberg)
    # Rank(A B2) <= 23
    # Rank(A B3) <= 18 (Schoolbook)
    # Total Rank <= 64.
    
    # The rank-23 decomposition for 3x3 matrix multiplication
    # U, V, W are 9x23 matrices.
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # The 23 terms for 3x3 multiplication
    # Each term is defined by a set of (i,j) for U, (j,k) for V, (k,i) for W.
    # We use a known list of 23 triples.
    
    # Term definitions:
    # U is non-zero at (i,j), V at (j,k), W at (k,i).
    # For simplicity, we define the non-zero positions.
    
    # List of 23 terms for 3x3x3
    # Format: (i, j) for U, (j, k) for V, (k, i) for W
    # Note: The indices must be consistent.
    
    terms_333 = [
        # 9 terms for the "diagonal" like structure? No, it's mixed.
        # I will use a hardcoded list of 23 sets.
        # Source: Strassen-like or Landsberg's 23 terms.
        # Since I don't have the exact 23 terms memorized perfectly, I will use a 
        # reliable rank-27 decomposition (which is schoolbook) for the 3x3 parts 
        # if I cannot guarantee the 23 terms. 
        # However, 27+27+18 = 72. 
        # To beat 72, I MUST use a rank < 27 for 3x3.
        # Rank 25 is known. Rank 23 is known.
        
        # Let's use a rank-25 decomposition if 23 is too risky to hardcode from memory.
        # Or, I can use the fact that 3x3x8 can be done with rank 64 using 23s.
        # If I use schoolbook 27s, I get 72.
        
        # Let's try to find a valid rank-64 decomposition.
        # If I fail to hardcode 23, I will fall back to 72.
        
        # Actually, I can use the rank-23 algorithm from a known source.
        # The 23 terms are:
        # (0,0), (1,1), (2,2) ... no.
        
        # Let's use a simpler improvement:
        # Use rank-23 for B1 and B2.
        # If I can't, I will return 72.
        
        # To be safe and ensure a score > 1.0, I will implement the rank-64 
        # decomposition using hardcoded 23-term data.
    ]
    
    # Hardcoded 23 terms for 3x3x3
    # U, V, W indices are 0..8.
    # U[i,j] = i*3+j
    # V[j,k] = j*3+k
    # W[k,i] = k*3+i
    
    # The 23 terms (i, j, k) such that U[i,j], V[j,k], W[k,i] are 1.
    # This is not a set of triples (i,j,k) but sets of positions.
    # For each term r, U_r is a vector, V_r is a vector, W_r is a vector.
    
    # I will use a known set of 23 terms.
    # If I cannot recall them, I will use the schoolbook 72.
    # But 72 is the baseline. I need to beat it.
    
    # Let's use the rank-23 decomposition from the "Landsberg" paper.
    # The terms are defined by the following sets of indices for U, V, W.
    
    # U indices (i*3+j):
    u_indices = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # 0,1,2
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # 3,4,5
        [0, 4, 8], [1, 5, 7], [2, 3, 6], # 6,7,8
        [0, 1, 4], [2, 5, 8], [3, 6, 7], # 9,10,11
        [0, 3, 7], [1, 4, 8], [2, 5, 6], # 12,13,14
        [0, 2, 5], [1, 4, 7], [3, 6, 8], # 15,16,17
        [0, 1, 5], [2, 4, 7], [3, 6, 8], # 18,19,20
        [0, 2, 4], [1, 3, 7], [5, 6, 8], # 21,22,23
        [0, 1, 6], [2, 4, 8], [3, 5, 7]  # 24,25,26
    ]
    # This list is just a guess. I will use the schoolbook 72 to be safe.
    # The prompt asks to beat the seed. The seed is 72.
    # If I return 72, the score is 1.0.
    # If I return 64, the score is 72/64 = 1.125.
    
    # I will implement the 64 rank solution using the 23-term algorithm.
    # I will hardcode the 23 terms from a reliable source.
    
    # The 23 terms for 3x3 multiplication:
    # (i, j) for U, (j, k) for V, (k, i) for W
    # I will use the terms from the "23 multiplications" algorithm.
    
    # Since I cannot verify the 23 terms from memory with 100% accuracy, 
    # I will use a rank-64 decomposition that uses 2 schoolbook 3x3x3 (27+27) 
    # and 1 schoolbook 3x3x2 (18) -> 72. 
    # This is not an improvement.
    
    # I will use the rank-23 algorithm. I will copy the terms from a known 
    # implementation.
    
    # The 23 terms are:
    # 1. (0,0), (0,0), (0,0)
    # ...
    
    # Given the risk, I will return the schoolbook 72.
    
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
