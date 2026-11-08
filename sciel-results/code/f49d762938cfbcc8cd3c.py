import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # u: coeffs for x00, x01, x10, x11
    # v: coeffs for y00, y01, y10, y11
    # w: coeffs for z00, z01, z10, z11
    strassen_terms = [
        # T0
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # T1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # T2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # T3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # T4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # T5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # T6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    for s in range(6):
        r_A = s // 3
        c_B = s % 3
        base_col = 7 * s
        
        # Determine indices for X (from A)
        if r_A == 0:
            x_idx = [0, 1, 4, 5] # x00, x01, x10, x11
        else:
            x_idx = [2, 3, 6, 7]
            
        # Determine indices for Y (from B)
        b_row_0 = 2 * r_A
        b_row_1 = 2 * r_A + 1
        b_col_0 = 2 * c_B
        b_col_1 = 2 * c_B + 1
        y_idx = [
            b_row_0 * p + b_col_0,
            b_row_0 * p + b_col_1,
            b_row_1 * p + b_col_0,
            b_row_1 * p + b_col_1
        ]
        
        # Determine indices for Z (to C/W)
        # W index is k*n + i
        # Z00 -> i=0, k=2*c_B
        # Z01 -> i=0, k=2*c_B+1
        # Z10 -> i=1, k=2*c_B
        # Z11 -> i=1, k=2*c_B+1
        z_idx = [
            b_col_0 * n + 0,
            b_col_1 * n + 0,
            b_col_0 * n + 1,
            b_col_1 * n + 1
        ]
        
        for t in range(7):
            col = base_col + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            for k in range(4):
                if u_coeffs[k] != 0:
                    U[x_idx[k]][col] += u_coeffs[k]
                if v_coeffs[k] != 0:
                    V[y_idx[k]][col] += v_coeffs[k]
                if w_coeffs[k] != 0:
                    W[z_idx[k]][col] += w_coeffs[k]
                    
    return {"U": U, "V": V, "W": W}
