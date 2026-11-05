import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block decomposition: 6x6 = 2x2 blocks of 3x3 matrices
    # Outer: 2x2 matrix mult using Strassen (rank 7)
    # Inner: 3x3 matrix mult using schoolbook (rank 27)
    # Total rank = 7 * 27 = 189
    
    block_size = 3
    total_rank = 189
    
    U = np.zeros((36, total_rank), dtype=np.float64)
    V = np.zeros((36, total_rank), dtype=np.float64)
    W = np.zeros((36, total_rank), dtype=np.float64)
    
    # Strassen's 7 steps for 2x2 matrix multiplication
    # A blocks: A[I,J] where I,J in {0,1}
    # B blocks: B[J,K] where J,K in {0,1}  
    # C blocks: C[I,K] where I,K in {0,1}
    
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        {'A': {(0,0): 1, (1,1): 1},
         'B': {(0,0): 1, (1,1): 1},
         'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10 + A11) * B00
        {'A': {(1,0): 1, (1,1): 1},
         'B': {(0,0): 1},
         'C': {(1,0): 1, (1,1): -1}},
        # M3 = A00 * (B01 - B11)
        {'A': {(0,0): 1},
         'B': {(0,1): 1, (1,1): -1},
         'C': {(0,1): 1, (1,1): 1}},
        # M4 = A11 * (B10 - B00)
        {'A': {(1,1): 1},
         'B': {(1,0): 1, (0,0): -1},
         'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00 + A01) * B11
        {'A': {(0,0): 1, (0,1): 1},
         'B': {(1,1): 1},
         'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10 - A00) * (B00 + B01)
        {'A': {(1,0): 1, (0,0): -1},
         'B': {(0,0): 1, (0,1): 1},
         'C': {(1,1): 1}},
        # M7 = (A01 - A11) * (B10 + B11)
        {'A': {(0,1): 1, (1,1): -1},
         'B': {(1,0): 1, (1,1): 1},
         'C': {(0,0): 1}}
    ]
    
    col = 0
    dim = 6
    
    for step in strassen_steps:
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        for r in range(block_size):   # local row
            for s in range(block_size):  # local sum index
                for c in range(block_size):  # local col
                    
                    # U: X[r,s] = sum_{I,J} A_coeffs[(I,J)] * A[I*3+r, J*3+s]
                    for (I, J), ca in A_coeffs.items():
                        if ca != 0:
                            i_global = I * block_size + r
                            j_global = J * block_size + s
                            U[i_global * dim + j_global, col] += ca
                    
                    # V: Y[s,c] = sum_{J,K} B_coeffs[(J,K)] * B[J*3+s, K*3+c]
                    for (J, K), cb in B_coeffs.items():
                        if cb != 0:
                            j_global = J * block_size + s
                            k_global = K * block_size + c
                            V[j_global * dim + k_global, col] += cb
                    
                    # W: M[r,c] contributes to C[I,K] with weight C_coeffs[(I,K)]
                    # Trace convention: W[k*n+i, r] where C[i,k] is the output
                    for (I, K), cc in C_coeffs.items():
                        if cc != 0:
                            i_global = I * block_size + r
                            k_global = K * block_size + c
                            W[k_global * dim + i_global, col] += cc
                    
                    col += 1
    
    # Safety check
    if time.time() - start_time > 140:
        pass  # Should not happen for this deterministic construction
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
