import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Returns a rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    The decomposition uses a Strassen-based approach where we split the 4x4 middle
    dimension into two 2x2 blocks, and the 8-dimensional output into four 2-dimensional blocks.
    
    U: shape (8, 56)  -- coefficients for A (2x4 = 8 entries)
    V: shape (32, 56) -- coefficients for B (4x8 = 32 entries)
    W: shape (16, 56) -- coefficients for C (8x2 = 16 entries), W[k*2+i, r]
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
    for all i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56
    
    col = 0
    
    # Strassen terms for 2x2x2 multiplication:
    # Local A indices: 00:0, 01:1, 10:2, 11:3
    # Local B indices: 00:0, 01:1, 10:2, 11:3
    # Local C indices: 00:0, 01:1, 10:2, 11:3
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = [
        # M1: (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # We have 4 output blocks (k=0,1; k=2,3; k=4,5; k=6,7)
    # and 2 input A blocks (j=0,1; j=2,3)
    # For each combination, we apply Strassen to a 2x2x2 subproblem.
    # Total: 4 * 2 * 7 = 56 terms.
    
    for block in range(4):  # output column block: cols 2*block, 2*block+1
        for a_half in range(2):  # A row half: rows 2*a_half, 2*a_half+1
            # A global indices for this 2x2 block:
            # A is 2x4, indexed as A[i, j] where i in {0,1}, j in {0,1,2,3}
            # Flat index: i*4 + j
            # For a_half=0: j in {0,1}, so indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
            # For a_half=1: j in {2,3}, so indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
            a_map = [a_half*2 + 0, a_half*2 + 1, 4 + a_half*2 + 0, 4 + a_half*2 + 1]
            # Wait, let me reconsider. A[i,j] flat = i*4+j.
            # For a_half=0 (j=0,1): 
            #   local 0 (A[0,0]) -> flat 0*4+0 = 0
            #   local 1 (A[0,1]) -> flat 0*4+1 = 1
            #   local 2 (A[1,0]) -> flat 1*4+0 = 4
            #   local 3 (A[1,1]) -> flat 1*4+1 = 5
            # For a_half=1 (j=2,3):
            #   local 0 (A[0,2]) -> flat 0*4+2 = 2
            #   local 1 (A[0,3]) -> flat 0*4+3 = 3
            #   local 2 (A[1,2]) -> flat 1*4+2 = 6
            #   local 3 (A[1,3]) -> flat 1*4+3 = 7
            a_map = [a_half * 2, a_half * 2 + 1, 4 + a_half * 2, 4 + a_half * 2 + 1]
            
            # B is 4x8, indexed as B[j, k] where j in {0,1,2,3}, k in {0,...,7}
            # Flat index: j*8 + k
            # For this subproblem, we use B rows 2*a_half, 2*a_half+1 and columns 2*block, 2*block+1
            # Local B indices:
            #   local 0 (B[2*a_half, 2*block]) -> flat (2*a_half)*8 + 2*block
            #   local 1 (B[2*a_half, 2*block+1]) -> flat (2*a_half)*8 + 2*block + 1
            #   local 2 (B[2*a_half+1, 2*block]) -> flat (2*a_half+1)*8 + 2*block
            #   local 3 (B[2*a_half+1, 2*block+1]) -> flat (2*a_half+1)*8 + 2*block + 1
            b_row_base = 2 * a_half
            b_col_base = 2 * block
            b_map = [
                b_row_base * 8 + b_col_base,
                b_row_base * 8 + b_col_base + 1,
                (b_row_base + 1) * 8 + b_col_base,
                (b_row_base + 1) * 8 + b_col_base + 1
            ]
            
            # C is 8x2, indexed as C[k, i] where k in {0,...,7}, i in {0,1}
            # Flat index: k*2 + i
            # For this subproblem, output columns are 2*block, 2*block+1
            # Local C indices:
            #   local 0 (C[2*block, 0]) -> flat (2*block)*2 + 0
            #   local 1 (C[2*block+1, 0]) -> flat (2*block+1)*2 + 0
            #   local 2 (C[2*block, 1]) -> flat (2*block)*2 + 1
            #   local 3 (C[2*block+1, 1]) -> flat (2*block+1)*2 + 1
            c_col_base = 2 * block
            c_map = [
                c_col_base * 2 + 0,
                (c_col_base + 1) * 2 + 0,
                c_col_base * 2 + 1,
                (c_col_base + 1) * 2 + 1
            ]
            
            for u_loc, v_loc, w_loc in terms:
                for l_idx, val in u_loc.items():
                    g_idx = a_map[l_idx]
                    U[g_idx][col] += val
                for l_idx, val in v_loc.items():
                    g_idx = b_map[l_idx]
                    V[g_idx][col] += val
                for l_idx, val in w_loc.items():
                    g_idx = c_map[l_idx]
                    W[g_idx][col] += val
                col += 1
    
    return {"U": U, "V": V, "W": W}
