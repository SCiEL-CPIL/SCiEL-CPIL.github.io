import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Strassen 2x2x2 base terms
    # Indices for 2x2 matrices: 00=0, 01=1, 10=2, 11=3
    strassen_2x2 = [
        # (u_indices, v_indices, w_indices)
        # P1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # P2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # P4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),
        # P5
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, -1)]),
        # P6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # P7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    
    all_terms = []
    
    # Part 1: 4x4x4 multiplication (first 4 columns of B)
    # We decompose 4x4x4 into 8 2x2x2 multiplications
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    
    # For each of the 8 block multiplications, we add 7 terms from Strassen 2x2
    # Block indices (r, c) for A, B, C blocks
    # A block (r, c): rows 2r..2r+1, cols 2c..2c+1
    # B block (r, c): rows 2r..2r+1, cols 2c..2c+1 (within first 4 cols)
    # C block (r, c): rows 2r..2r+1, cols 2c..2c+1 (within first 4 cols)
    
    # Define the 8 block multiplications: (A_block_idx, B_block_idx, C_block_idx)
    # A_block_idx: (r, c)
    # B_block_idx: (r, c)
    # C_block_idx: (r, c)
    
    block_mults = [
        ((0,0), (0,0), (0,0)), # A00 B00 -> C00
        ((0,1), (1,0), (0,0)), # A01 B10 -> C00
        ((0,0), (0,1), (0,1)), # A00 B01 -> C01
        ((0,1), (1,1), (0,1)), # A01 B11 -> C01
        ((1,0), (0,0), (1,0)), # A10 B00 -> C10
        ((1,1), (1,0), (1,0)), # A11 B10 -> C10
        ((1,0), (0,1), (1,1)), # A10 B01 -> C11
        ((1,1), (1,1), (1,1)), # A11 B11 -> C11
    ]
    
    for (ar, ac), (br, bc), (cr, cc) in block_mults:
        for u_s, v_s, w_s in strassen_2x2:
            u_global = []
            v_global = []
            w_global = []
            
            # Map local 2x2 indices to global 4x4 (A) and 4x5 (B, C) indices
            for local_idx, coeff in u_s:
                i, j = divmod(local_idx, 2) # i is row, j is col in 2x2
                # A global index: (2*ar + i) * 4 + (2*ac + j)
                g_idx = (2*ar + i) * 4 + (2*ac + j)
                u_global.append((g_idx, coeff))
                
            for local_idx, coeff in v_s:
                i, j = divmod(local_idx, 2)
                # B global index: (2*br + i) * 5 + (2*bc + j)
                # Note: B is 4x5, but we are using the first 4 columns, so col index is 2*bc+j
                g_idx = (2*br + i) * 5 + (2*bc + j)
                v_global.append((g_idx, coeff))
                
            for local_idx, coeff in w_s:
                i, j = divmod(local_idx, 2)
                # C global index in W: k*n + i
                # C row = 2*cr + i
                # C col = 2*cc + j
                c_row = 2*cr + i
                c_col = 2*cc + j
                w_idx = c_col * n + c_row
                w_global.append((w_idx, coeff))
                
            all_terms.append((u_global, v_global, w_global))
            
    # Part 2: 4x4x1 multiplication (5th column of B)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(4):
        for j in range(4):
            u = [(i * 4 + j, 1)]
            v = [(j * 5 + 4, 1)]
            # C[i, 4] -> W index 4*4 + i
            w = [(4 * 4 + i, 1)]
            all_terms.append((u, v, w))
            
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for idx, coeff in u:
            U[idx][r] = coeff
        for idx, coeff in v:
            V[idx][r] = coeff
        for idx, coeff in w:
            W[idx][r] = coeff
            
    return {"U": U, "V": V, "W": W}
