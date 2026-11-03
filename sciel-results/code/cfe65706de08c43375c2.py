import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Rank calculation:
    # 3 column blocks of B (each 4x2)
    # Each block multiplication is 2x4 * 4x2 -> 2x2
    # Split 2x4 A into two 2x2 blocks
    # So each block mult is sum of two 2x2x2 mults
    # Total 2x2x2 mults = 3 * 2 = 6
    # Rank of 2x2x2 (Strassen) = 7
    # Total Rank = 6 * 7 = 42
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Coefficients for A (a,b,c,d)
    strassen_A = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 1, 0, 0], # m5
        [0, 0, 1, -1], # m6
        [0, 1, 0, -1]  # m7
    ]
    
    # Coefficients for B (e,f,g,h)
    strassen_B = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 1, 0, -1], # m3
        [0, 0, 1, -1], # m4
        [0, 0, 0, 1], # m5
        [1, 1, 0, 0], # m6
        [0, 0, 1, 1]  # m7
    ]
    
    # Reconstruction of C (c00, c01, c10, c11) from m1..m7
    # C00 = m1 - m4 + m5 + m6
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 + m7
    strassen_C = [
        [1, 0, 0, -1, 1, 1, 0], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, 0, -1, 0, 0, 0, 1]  # C11
    ]
    
    # We have 6 instances of 2x2x2 multiplication
    # Instance defined by (col_block, a_split)
    # col_block in 0, 1, 2 (columns 0-1, 2-3, 4-5 of B)
    # a_split in 0, 1 (Left/Right of A, which corresponds to Top/Bottom of B_block)
    
    current_rank = 0
    for col_block in range(3):
        for a_split in range(2):
            # Determine global indices for A, B, C parts
            
            # A is 2x4. 
            # If a_split=0 (Left), A uses cols 0,1. Indices: 0,1,4,5 (row-major)
            # If a_split=1 (Right), A uses cols 2,3. Indices: 2,3,6,7
            if a_split == 0:
                a_indices = [0, 1, 4, 5] # a, b, c, d
            else:
                a_indices = [2, 3, 6, 7] # a, b, c, d
            
            # B is 4x6.
            # col_block determines columns 2*col_block, 2*col_block+1
            # If a_split=0 (Top), B uses rows 0,1.
            # If a_split=1 (Bottom), B uses rows 2,3.
            b_col_start = 2 * col_block
            if a_split == 0:
                # Rows 0, 1. Cols b_col_start, b_col_start+1
                # B indices: row*6 + col
                b_indices = [
                    0 * 6 + b_col_start,
                    0 * 6 + b_col_start + 1,
                    1 * 6 + b_col_start,
                    1 * 6 + b_col_start + 1
                ] # e, f, g, h
            else:
                # Rows 2, 3. Cols b_col_start, b_col_start+1
                b_indices = [
                    2 * 6 + b_col_start,
                    2 * 6 + b_col_start + 1,
                    3 * 6 + b_col_start,
                    3 * 6 + b_col_start + 1
                ] # e, f, g, h
            
            # C is 2x6.
            # Block is at columns b_col_start, b_col_start+1.
            # C indices in W: W[k*n + i] where n=2.
            # C[0, b_col_start] -> k=b_col_start, i=0 -> idx = b_col_start*2 + 0
            # C[0, b_col_start+1] -> k=b_col_start+1, i=0 -> idx = (b_col_start+1)*2 + 0
            # C[1, b_col_start] -> k=b_col_start, i=1 -> idx = b_col_start*2 + 1
            # C[1, b_col_start+1] -> k=b_col_start+1, i=1 -> idx = (b_col_start+1)*2 + 1
            
            c_indices = [
                b_col_start * 2 + 0,      # C00
                (b_col_start + 1) * 2 + 0, # C01
                b_col_start * 2 + 1,      # C10
                (b_col_start + 1) * 2 + 1  # C11
            ]
            
            # Fill U, V, W for the 7 Strassen terms
            for s in range(7):
                r = current_rank
                # U: coefficients for A
                for idx, val in zip(a_indices, strassen_A[s]):
                    U[idx, r] = val
                
                # V: coefficients for B
                for idx, val in zip(b_indices, strassen_B[s]):
                    V[idx, r] = val
                
                # W: coefficients for C
                for idx, val in zip(c_indices, strassen_C[:, s]):
                    W[idx, r] = val
                
                current_rank += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
