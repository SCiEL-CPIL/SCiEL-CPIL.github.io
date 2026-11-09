import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # 4 blocks of Strassen
    # Each block rank 7
    total_rank = 28
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U factors (A)
    # M1: 1,0, 0,1 (A00, A11)
    # M2: 0,1, 0,1 (A10, A11)
    # M3: 1,0, 0,0 (A00)
    # M4: 0,0, 0,1 (A11)
    # M5: 1,1, 0,0 (A00, A01)
    # M6: 0,1, -1,0 (A10, -A00)
    # M7: 0,1, 0,-1 (A01, -A11)
    
    U_strassen = [
        [1, 0, 0, 1], # M1: A00, A11
        [0, 1, 0, 1], # M2: A10, A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00, A01
        [0, 1, -1, 0],# M6: A10, -A00
        [0, 1, 0, -1] # M7: A01, -A11
    ]
    
    # V factors (B)
    # M1: 1,0, 0,1 (B00, B11)
    # M2: 1,0, 0,0 (B00)
    # M3: 0,1, 0,-1 (B01, -B11)
    # M4: 0,-1, 1,0 (B10, -B00)
    # M5: 0,0, 0,1 (B11)
    # M6: 1,1, 0,0 (B00, B01)
    # M7: 0,1, 1,0 (B10, B11)
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, -1, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 1, 1, 0]  # M7
    ]
    
    # W factors (C)
    # C indices: 00, 01, 10, 11
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    W_strassen = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Blocks:
    # 0: A1, B1a -> C1a (A cols 0-1, B rows 0-1, C cols 0-1)
    # 1: A1, B1b -> C1b (A cols 0-1, B rows 0-1, C cols 2-3)
    # 2: A2, B2a -> C2a (A cols 2-3, B rows 2-3, C cols 0-1)
    # 3: A2, B2b -> C2b (A cols 2-3, B rows 2-3, C cols 2-3)
    
    blocks = [
        (0, 0, 0), # A_j_offset, B_j_offset, C_k_offset
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2)
    ]
    
    for b_idx, (a_off, b_off, c_off) in enumerate(blocks):
        for m_idx in range(7):
            col = b_idx * 7 + m_idx
            
            # U
            u_vals = U_strassen[m_idx]
            # u_vals[0] is A00, u_vals[1] is A01, u_vals[2] is A10, u_vals[3] is A11
            # A index: i * 4 + j
            # i=0: j=0,1
            # i=1: j=2,3
            # A00: i=0, j=0+a_off
            # A01: i=0, j=1+a_off
            # A10: i=1, j=0+a_off
            # A11: i=1, j=1+a_off
            
            U[0 * 4 + (0 + a_off)][col] += u_vals[0]
            U[0 * 4 + (1 + a_off)][col] += u_vals[1]
            U[1 * 4 + (0 + a_off)][col] += u_vals[2]
            U[1 * 4 + (1 + a_off)][col] += u_vals[3]
            
            # V
            v_vals = V_strassen[m_idx]
            # B index: j * 4 + k
            # B00: j=0+b_off, k=0+c_off
            # B01: j=0+b_off, k=1+c_off
            # B10: j=1+b_off, k=0+c_off
            # B11: j=1+b_off, k=1+c_off
            
            V[(0 + b_off) * 4 + (0 + c_off)][col] += v_vals[0]
            V[(0 + b_off) * 4 + (1 + c_off)][col] += v_vals[1]
            V[(1 + b_off) * 4 + (0 + c_off)][col] += v_vals[2]
            V[(1 + b_off) * 4 + (1 + c_off)][col] += v_vals[3]
            
            # W
            w_vals = W_strassen[m_idx]
            # C index: k * 2 + i
            # C00: i=0, k=0+c_off
            # C01: i=0, k=1+c_off
            # C10: i=1, k=0+c_off
            # C11: i=1, k=1+c_off
            
            W[(0 + c_off) * 2 + 0][col] += w_vals[0]
            W[(1 + c_off) * 2 + 0][col] += w_vals[1]
            W[(0 + c_off) * 2 + 1][col] += w_vals[2]
            W[(1 + c_off) * 2 + 1][col] += w_vals[3]
            
    return {"U": U, "V": V, "W": W}
