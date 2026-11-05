import time
import numpy as np

def construct():
    n, m, p = 3, 4, 7
    rank = 24
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: A2 (row 2) * B1 (cols 0-3) -> C (row 2, cols 0-3)
    # i=2, j in 0..3, k in 0..3
    # Rank 4: r in 0..3
    for r in range(4):
        j = r
        # U[2*m + j, r] = 1
        U[2 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 0..3
        for k in range(4):
            V[j * p + k, r] = 1.0
        # W[k*n + 2, r] = 1 for k in 0..3
        for k in range(4):
            W[k * n + 2, r] = 1.0
            
    # Block 2: A2 (row 2) * B2 (cols 4-6) -> C (row 2, cols 4-6)
    # i=2, j in 0..3, k in 4..6
    # Rank 4: r in 4..7
    for r in range(4, 8):
        j = r - 4
        # U[2*m + j, r] = 1
        U[2 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 4..6
        for k in range(4, 7):
            V[j * p + k, r] = 1.0
        # W[k*n + 2, r] = 1 for k in 4..6
        for k in range(4, 7):
            W[k * n + 2, r] = 1.0

    # Block 3: A1 (rows 0-1) * B1 (cols 0-3) -> C (rows 0-1, cols 0-3)
    # i in 0..1, j in 0..3, k in 0..3
    # Rank 8: r in 8..15
    # Sub-block i=0: r in 8..11
    for r in range(8, 12):
        j = r - 8
        # U[0*m + j, r] = 1
        U[0 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 0..3
        for k in range(4):
            V[j * p + k, r] = 1.0
        # W[k*n + 0, r] = 1 for k in 0..3
        for k in range(4):
            W[k * n + 0, r] = 1.0
            
    # Sub-block i=1: r in 12..15
    for r in range(12, 16):
        j = r - 12
        # U[1*m + j, r] = 1
        U[1 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 0..3
        for k in range(4):
            V[j * p + k, r] = 1.0
        # W[k*n + 1, r] = 1 for k in 0..3
        for k in range(4):
            W[k * n + 1, r] = 1.0

    # Block 4: A1 (rows 0-1) * B2 (cols 4-6) -> C (rows 0-1, cols 4-6)
    # i in 0..1, j in 0..3, k in 4..6
    # Rank 8: r in 16..23
    # Sub-block i=0: r in 16..19
    for r in range(16, 20):
        j = r - 16
        # U[0*m + j, r] = 1
        U[0 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 4..6
        for k in range(4, 7):
            V[j * p + k, r] = 1.0
        # W[k*n + 0, r] = 1 for k in 4..6
        for k in range(4, 7):
            W[k * n + 0, r] = 1.0
            
    # Sub-block i=1: r in 20..23
    for r in range(20, 24):
        j = r - 20
        # U[1*m + j, r] = 1
        U[1 * m + j, r] = 1.0
        # V[j*p + k, r] = 1 for k in 4..6
        for k in range(4, 7):
            V[j * p + k, r] = 1.0
        # W[k*n + 1, r] = 1 for k in 4..6
        for k in range(4, 7):
            W[k * n + 1, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
