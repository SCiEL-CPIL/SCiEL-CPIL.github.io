import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 98 solution
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen vectors for 2x2x2
    # U_in, V_in, W_in are lists of 7 vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_in = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, -1, 0] # M7: b-d
    ]
    V_in = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    W_in = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 0, 1, -1],# M2: C10+1, C11-1
        [0, 1, 0, 1], # M3: C01+1, C11+1
        [1, 0, 1, 0], # M4: C00+1, C10+1
        [-1, 1, 0, 0],# M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [1, 0, 0, 0]  # M7: C00+1
    ]
    
    # Outer Strassen definitions for 4x4x4
    # Each entry: (A_blocks, B_blocks, C_blocks, C_coeff)
    # A_blocks: dict mapping block name to coefficient
    # B_blocks: dict mapping block name to coefficient
    # C_blocks: dict mapping block name to coefficient
    
    # S1 = (A11+A22)(B11+B22) -> C11+1, C22+1
    # S2 = (A21+A22)B11 -> C21+1, C22-1
    # S3 = A11(B12-B22) -> C12+1, C22+1
    # S4 = A22(B21-B11) -> C11+1, C21+1
    # S5 = (A11+A12)B22 -> C12+1, C11-1
    # S6 = (A21-A11)(B11+B12) -> C22+1
    # S7 = (A12-A22)(B21+B22) -> C11+1
    
    outer_terms = [
        # S1
        (
            {'11': 1, '22': 1}, 
            {'11': 1, '22': 1}, 
            {'11': 1, '22': 1}
        ),
        # S2
        (
            {'21': 1, '22': 1}, 
            {'11': 1}, 
            {'21': 1, '22': -1}
        ),
        # S3
        (
            {'11': 1}, 
            {'12': 1, '22': -1}, 
            {'12': 1, '22': 1}
        ),
        # S4
        (
            {'22': 1}, 
            {'21': 1, '11': -1}, 
            {'11': 1, '21': 1}
        ),
        # S5
        (
            {'11': 1, '12': 1}, 
            {'22': 1}, 
            {'12': 1, '11': -1}
        ),
        # S6
        (
            {'21': 1, '11': -1}, 
            {'11': 1, '12': 1}, 
            {'22': 1}
        ),
        # S7
        (
            {'12': 1, '22': -1}, 
            {'21': 1, '22': 1}, 
            {'11': 1}
        )
    ]
    
    # Block index mappings for A (4x4)
    # Block '11': rows 0-1, cols 0-1. Indices: 0,1,4,5
    # Block '12': rows 0-1, cols 2-3. Indices: 2,3,6,7
    # Block '21': rows 2-3, cols 0-1. Indices: 8,9,12,13
    # Block '22': rows 2-3, cols 2-3. Indices: 10,11,14,15
    
    A_block_indices = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15]
    }
    
    # Block index mappings for B (4x8)
    # Left Half (cols 0-3):
    # '11': rows 0-1, cols 0-1. Indices: 0,1,8,9
    # '12': rows 0-1, cols 2-3. Indices: 2,3,10,11
    # '21': rows 2-3, cols 0-1. Indices: 16,17,24,25
    # '22': rows 2-3, cols 2-3. Indices: 18,19,26,27
    
    # Right Half (cols 4-7):
    # '11': rows 0-1, cols 4-5. Indices: 4,5,12,13
    # '12': rows 0-1, cols 6-7. Indices: 6,7,14,15
    # '21': rows 2-3, cols 4-5. Indices: 20,21,28,29
    # '22': rows 2-3, cols 6-7. Indices: 22,23,30,31
    
    B_block_indices_L = {
        '11': [0, 1, 8, 9],
        '12': [2, 3, 10, 11],
        '21': [16, 17, 24, 25],
        '22': [18, 19, 26, 27]
    }
    B_block_indices_R = {
        '11': [4, 5, 12, 13],
        '12': [6, 7, 14, 15],
        '21': [20, 21, 28, 29],
        '22': [22, 23, 30, 31]
    }
    
    # Block index mappings for C (4x8)
    # C is indexed by k*n + i. n=4.
    # Left Half (cols 0-3 of C, i.e., k=0..3? No. C is 4x8. Rows 0-3, Cols 0-7).
    # Prompt: W[k*n + i]. k is row of C? No.
    # C = A B. A is 4x4. B is 4x8. C is 4x8.
    # Standard C[i, k].
    # Prompt says W[k*n + i]. This implies the flat index is k*4 + i.
    # This corresponds to C^T stored in row-major, or C stored in column-major.
    # Let's assume k is the column index of C (0..7) and i is the row index of C (0..3).
    # Wait, if k goes 0..7 and i goes 0..3, then k*n+i ranges 0..31.
    # This matches p*n = 8*4 = 32.
    # So W index = k*4 + i.
    # This means for a fixed k (column of C), the 4 entries for i=0..3 are contiguous.
    
    # Left Half (C cols 0-3):
    # C11 (rows 0-1, cols 0-1): k in 0,1. i in 0,1.
    # Indices: k=0: 0,1. k=1: 4,5. -> [0,1,4,5]
    # C12 (rows 0-1, cols 2-3): k in 2,3. i in 0,1.
    # Indices: k=2: 8,9. k=3: 12,13. -> [8,9,12,13]
    # C21 (rows 2-3, cols 0-1): k in 0,1. i in 2,3.
    # Indices: k=0: 2,3. k=1: 6,7. -> [2,3,6,7]
    # C22 (rows 2-3, cols 2-3): k in 2,3. i in 2,3.
    # Indices: k=2: 10,11. k=3: 14,15. -> [10,11,14,15]
    
    # Right Half (C cols 4-7):
    # C11 (rows 0-1, cols 4-5): k in 4,5. i in 0,1.
    # Indices: k=4: 16,17. k=5: 20,21. -> [16,17,20,21]
    # C12 (rows 0-1, cols 6-7): k in 6,7. i in 0,1.
    # Indices: k=6: 24,25. k=7: 28,29. -> [24,25,28,29]
    # C21 (rows 2-3, cols 4-5): k in 4,5. i in 2,3.
    # Indices: k=4: 18,19. k=5: 22,23. -> [18,19,22,23]
    # C22 (rows 2-3, cols 6-7): k in 6,7. i in 2,3.
    # Indices: k=6: 26,27. k=7: 30,31. -> [26,27,30,31]
    
    C_block_indices_L = {
        '11': [0, 1, 4, 5],
        '12': [8, 9, 12, 13],
        '21': [2, 3, 6, 7],
        '22': [10, 11, 14, 15]
    }
    C_block_indices_R = {
        '11': [16, 17, 20, 21],
        '12': [24, 25, 28, 29],
        '21': [18, 19, 22, 23],
        '22': [26, 27, 30, 31]
    }
    
    col = 0
    for half in ['L', 'R']:
        B_blocks = B_block_indices_L if half == 'L' else B_block_indices_R
        C_blocks = C_block_indices_L if half == 'L' else C_block_indices_R
        
        for outer_idx, (A_comb, B_comb, C_comb) in enumerate(outer_terms):
            for inner_idx in range(7):
                # Construct U
                U_vec = [0] * 16
                for block_name, coeff in A_comb.items():
                    indices = A_block_indices[block_name]
                    for j in range(4):
                        U_vec[indices[j]] += coeff * U_in[inner_idx][j]
                
                # Construct V
                V_vec = [0] * 32
                for block_name, coeff in B_comb.items():
                    indices = B_blocks[block_name]
                    for j in range(4):
                        V_vec[indices[j]] += coeff * V_in[inner_idx][j]
                
                # Construct W
                W_vec = [0] * 32
                for block_name, coeff in C_comb.items():
                    indices = C_blocks[block_name]
                    for j in range(4):
                        W_vec[indices[j]] += coeff * W_in[inner_idx][j]
                
                # Store
                for r in range(16):
                    U[r][col] = U_vec[r]
                for r in range(32):
                    V[r][col] = V_vec[r]
                    W[r][col] = W_vec[r]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
