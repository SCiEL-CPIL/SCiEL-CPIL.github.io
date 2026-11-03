import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 56 construction
    # 4 Strassen blocks (rank 7 each) + 28 schoolbook terms (rank 1 each)
    R = 56
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Define the 4 blocks
    # I: {0, 1}
    # J: {0, 1} and {2, 3}
    # K: {0, 1} and {2, 3}
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3])
    ]
    
    # Strassen coefficients for U, V, W for the 7 terms
    # Term 0: M1
    # Term 1: M2
    # Term 2: M3
    # Term 3: M4
    # Term 4: M5
    # Term 5: M6
    # Term 6: M7
    
    # Local indices 0, 1
    # U (i,j)
    U_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    
    # V (j,k)
    V_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    
    # W (k,i)
    W_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1 (C00, C11)
        {(0,1): 1, (1,1): -1},      # M2 (C10, C11)
        {(1,0): 1, (1,1): 1},       # M3 (C01, C11)
        {(0,0): 1, (0,1): 1},       # M4 (C00, C10)
        {(0,0): -1, (1,0): 1},      # M5 (C00, C01)
        {(1,1): 1},                 # M6 (C11)
        {(0,0): 1}                  # M7 (C00)
    ]
    
    # Add Strassen blocks
    for I, J, K in blocks:
        for t in range(7):
            # U
            for (li, lj), val in U_coeffs[t].items():
                i = I[li]
                j = J[lj]
                U[i * m + j, col] += val
            
            # V
            for (lj, lk), val in V_coeffs[t].items():
                j = J[lj]
                k = K[lk]
                V[j * p + k, col] += val
            
            # W
            for (lk, li), val in W_coeffs[t].items():
                k = K[lk]
                i = I[li]
                W[k * n + i, col] += val
            
            col += 1
    
    # Add schoolbook terms for the rest
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining:
    # 1. i=2, any j, any k
    # 2. i in {0,1}, any j, k=4
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3]:
                    continue
                
                # Schoolbook term
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify col == R
    assert col == R, f"Column count {col} != Rank {R}"
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
