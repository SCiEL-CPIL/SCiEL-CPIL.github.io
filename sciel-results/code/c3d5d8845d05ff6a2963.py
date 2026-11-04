import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 for 4x4x4 (Strassen) + 16 for 4x4x1 (Schoolbook)
    rank = 65
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec, idx):
        for i in range(len(u_vec)):
            if u_vec[i] != 0:
                U[i][idx] += u_vec[i]
        for i in range(len(v_vec)):
            if v_vec[i] != 0:
                V[i][idx] += v_vec[i]
        for i in range(len(w_vec)):
            if w_vec[i] != 0:
                W[i][idx] += w_vec[i]
                
    # Part 1: 4x4x4 using Strassen
    # Strassen terms for 2x2
    # u, v, w are length 4
    s_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Top level Strassen for 4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Coefficients for A blocks (4 blocks, each 2x2 -> 16 entries)
    # A11: 0,1,4,5; A12: 2,3,6,7; A21: 8,9,12,13; A22: 10,11,14,15
    # B11: 0,1,4,5; B12: 2,3,6,7; B21: 8,9,12,13; B22: 10,11,14,15
    # C11: 0,1,4,5; C12: 2,3,6,7; C21: 8,9,12,13; C22: 10,11,14,15
    # Note: B is 4x5. B11 is cols 0,1. B12 is cols 2,3.
    
    top_u = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    top_v = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    
    # C contributions
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    top_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    col_idx = 0
    for k in range(7):
        # Expand top level term k into 7 bottom level terms
        for j in range(7):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20 # B is 4x5, but we only use first 4 cols
            w_vec = [0.0] * 20 # C is 5x4, so 20 entries. k*n+i.
            
            # u_vec: combine top_u[k] with s_u[j]
            # top_u[k] gives coeffs for A11, A12, A21, A22
            # s_u[j] gives coeffs for the 4 entries of the 2x2 block
            for block_idx in range(4):
                coeff_top = top_u[k][block_idx]
                if coeff_top == 0: continue
                # Map block_idx to A indices
                # 0: A11 (0,1,4,5)
                # 1: A12 (2,3,6,7)
                # 2: A21 (8,9,12,13)
                # 3: A22 (10,11,14,15)
                offsets = [0, 1, 4, 5] if block_idx == 0 else \
                          [2, 3, 6, 7] if block_idx == 1 else \
                          [8, 9, 12, 13] if block_idx == 2 else \
                          [10, 11, 14, 15]
                for entry_idx in range(4):
                    u_vec[offsets[entry_idx]] += coeff_top * s_u[j][entry_idx]
            
            # v_vec: combine top_v[k] with s_v[j]
            # B blocks: B11 (0,1,4,5), B12 (2,3,6,7), B21 (8,9,12,13), B22 (10,11,14,15)
            for block_idx in range(4):
                coeff_top = top_v[k][block_idx]
                if coeff_top == 0: continue
                offsets = [0, 1, 4, 5] if block_idx == 0 else \
                          [2, 3, 6, 7] if block_idx == 1 else \
                          [8, 9, 12, 13] if block_idx == 2 else \
                          [10, 11, 14, 15]
                for entry_idx in range(4):
                    v_vec[offsets[entry_idx]] += coeff_top * s_v[j][entry_idx]
            
            # w_vec: combine top_w[k] with s_w[j]
            # C blocks: C11 (0,1,4,5), C12 (2,3,6,7), C21 (8,9,12,13), C22 (10,11,14,15)
            # Note: W index is k*n+i. C is 4x5.
            # C11 is rows 0,1 cols 0,1.
            # C12 is rows 0,1 cols 2,3.
            # C21 is rows 2,3 cols 0,1.
            # C22 is rows 2,3 cols 2,3.
            for block_idx in range(4):
                coeff_top = top_w[k][block_idx]
                if coeff_top == 0: continue
                offsets = [0, 1, 4, 5] if block_idx == 0 else \
                          [2, 3, 6, 7] if block_idx == 1 else \
                          [8, 9, 12, 13] if block_idx == 2 else \
                          [10, 11, 14, 15]
                for entry_idx in range(4):
                    w_vec[offsets[entry_idx]] += coeff_top * s_w[j][entry_idx]
                    
            add_term(u_vec, v_vec, w_vec, col_idx)
            col_idx += 1
            
    # Part 2: 4x4x1 (Column 5)
    # C[:, 4] = A @ B[:, 4]
    for i in range(4):
        for j in range(4):
            u_vec = [0.0] * 16
            v_vec = [0.0] * 20
            w_vec = [0.0] * 20
            
            u_vec[i * 4 + j] = 1.0
            v_vec[j * 5 + 4] = 1.0 # B index: row j, col 4. j*5+4
            w_vec[4 * 4 + i] = 1.0 # W index: k*n+i. k=4 (col 5 of C), i=row.
            # Wait, W[k*n+i, r]. k is col of C, i is row of C.
            # C is 4x5. k in 0..4, i in 0..3.
            # So index is k*4 + i.
            # For col 4 (5th col), k=4.
            # So index is 4*4 + i = 16 + i.
            
            add_term(u_vec, v_vec, w_vec, col_idx)
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
