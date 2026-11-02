import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    dim_A = n * m   # 8
    dim_B = m * p   # 24
    dim_C = p * n   # 12
    
    rank = 42
    
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    
    # Strassen 2x2x2 rank-7 decomposition
    # Local indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # U factors (A-side linear combinations)
    U_local = [
        [1, 0, 0, 1],   # P1: a00 + a11
        [0, 0, 1, 1],   # P2: a10 + a11
        [1, 0, 0, 0],   # P3: a00
        [0, 0, 0, 1],   # P4: a11
        [1, 1, 0, 0],   # P5: a00 + a01
        [-1, 0, 1, 0],  # P6: a10 - a00
        [0, 1, 0, -1]   # P7: a01 - a11
    ]
    
    # V factors (B-side linear combinations)
    V_local = [
        [1, 0, 0, 1],   # P1: b00 + b11
        [1, 0, 0, 0],   # P2: b00
        [0, 1, 0, -1],  # P3: b01 - b11
        [-1, 0, 1, 0],  # P4: b10 - b00
        [0, 0, 0, 1],   # P5: b11
        [1, 1, 0, 0],   # P6: b00 + b01
        [0, 0, 1, 1]    # P7: b10 + b11
    ]
    
    # W factors (C-side contributions)
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    W_local = [
        [1, 0, 0, 1],   # P1: c00 + c11
        [0, 0, 1, -1],  # P2: c10 - c11
        [0, 1, 0, 1],   # P3: c01 + c11
        [1, 0, 1, 0],   # P4: c00 + c10
        [-1, 1, 0, 0],  # P5: -c00 + c01
        [0, 0, 0, 1],   # P6: c11
        [1, 0, 0, 0]    # P7: c00
    ]
    
    col_offset = 0
    
    for cb in range(3):
        for side in ['L', 'R']:
            for t in range(7):
                r_col = col_offset
                
                # U mapping
                if side == 'L':
                    # A columns 0,1
                    a_col_base = 0
                else:
                    # A columns 2,3
                    a_col_base = 2
                
                # V mapping
                if side == 'L':
                    # B rows 0,1
                    b_row_base = 0
                else:
                    # B rows 2,3
                    b_row_base = 2
                
                # W mapping: C columns are 2*cb, 2*cb+1
                c_col_base = 2 * cb
                
                # Fill U
                for lr in range(2):
                    for lc in range(2):
                        val = U_local[t][lr * 2 + lc]
                        if val != 0:
                            ga = lr * m + (a_col_base + lc)
                            U[ga][r_col] += val
                
                # Fill V
                for lr in range(2):
                    for lc in range(2):
                        val = V_local[t][lr * 2 + lc]
                        if val != 0:
                            gb = (b_row_base + lr) * p + (c_col_base + lc)
                            V[gb][r_col] += val
                
                # Fill W
                for lr in range(2):
                    for lc in range(2):
                        val = W_local[t][lr * 2 + lc]
                        if val != 0:
                            # W index: k*n + i where k=col, i=row
                            gc = (c_col_base + lc) * n + lr
                            W[gc][r_col] += val
                
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
