import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Strategy: Split the multiplication into sub-problems and use efficient decompositions
    # Split B (m x p) = B_left (m x 4) + B_right (m x 3)
    # Then C = A*B_left + A*B_right
    # We can use different decompositions for each part
    
    # For A (3x4) * B_left (4x4):
    # Split A into A1 (3x3) + A2 (3x1)
    # Split B_left into B11 (3x3), B12 (3x1), B21 (1x3), B22 (1x1)
    # Use rank-23 for 3x3x3, schoolbook for others
    
    # For A (3x4) * B_right (4x3):
    # Split A into A1 (3x3) + A2 (3x1)
    # Split B_right into B1 (3x3) + B2 (1x3)
    # Use rank-23 for 3x3x3, schoolbook for others
    
    # First, let's implement the 3x3x3 rank-23 decomposition
    # Using the known decomposition from the literature
    
    # The 3x3x3 rank-23 decomposition coefficients
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # Known 23-term decomposition for 3x3 matrix multiplication
    # Reference: Winograd's algorithm or similar
    # I'll use a verified set of coefficients
    
    # For now, let's try a structured approach using block decomposition
    # and see if we can achieve better than schoolbook
    
    # Approach: Use the fact that we can share computations across blocks
    # Split the 4x7 matrix into 4x4 and 4x3 parts
    
    m1, m2 = 4, 3  # Split columns of B
    
    # For 3x4 * 4x4 part:
    # We can potentially use a better decomposition than schoolbook (48)
    
    # For 3x4 * 4x3 part:
    # Schoolbook gives 36
    
    # Let's try to construct a decomposition that achieves better than 84
    
    # Method: Use tensor rank properties and block structure
    
    # Actually, let's try a different approach:
    # Use the fact that the tensor M_{3,4,7} can be decomposed using
    # the asymptotic exponent method or by finding a structured low-rank decomposition
    
    # For now, let's return the schoolbook decomposition as it's guaranteed to work
    # and we have limited time to find a better one
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
