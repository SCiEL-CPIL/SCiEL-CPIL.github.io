import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build Rank 55 Decomposition
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Indices for A (2x3): a00, a01, a10, a11, a02, a12
    a00, a01, a10, a11, a02, a12 = 0, 1, 3, 4, 2, 5
    
    # 5 Strassen blocks for 2x2 submatrices
    # We have 3 columns in A and 10 in B.
    # We can form 5 disjoint 2x2 blocks?
    # A has 3 cols. B has 10 cols.
    # Block 1: A[:, 0:2], B[0:2, :] -> 2x2 x 2x2? No.
    # The previous code used 5 blocks of 2x2 Strassen.
    # This implies it treated the 3x10 part as 5 pairs of columns?
    # 3x10 -> 5 pairs of 2x2? No.
    # The previous code logic:
    # for b in range(5):
    #   k0 = 2*b, k1 = 2*b+1
    #   This uses 10 columns of B.
    #   And it used A indices a00, a11, a10, a01, a02, a12.
    #   It seems to map the 3 columns of A to specific roles.
    
    # I will replicate the exact logic of the previous working code to ensure validity.
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen 7 terms
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
    # Remaining 20 terms for the 3rd column of A?
    # The previous code added 20 terms.
    # a02, a12 are the 3rd column indices.
    for k in range(p):
        b2k = 2 * p + k
        U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
