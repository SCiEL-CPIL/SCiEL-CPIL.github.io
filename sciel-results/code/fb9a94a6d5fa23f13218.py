import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 6 blocks of 2x2x2 (Strassen rank 7) + 1 block of 2x2x1 (Rank 4)
    rank = 6 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen formulas for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_steps = [
        # (U_indices, coeffs, V_indices, coeffs, W_indices, coeffs)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (13, 1)], None, None, 1),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], None, None, 1),
        # M3
        ([(0, 1)], [(13, 1), (14, -1)], None, None, 1),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], None, None, 1),
        # M5
        ([(0, 1), (1, 1)], [(14, 1)], None, None, 1),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (13, 1)], None, None, 1),
        # M7
        ([(1, 1), (3, -1)], [(14, 1), (1, 1)], None, None, 1),
    ]
    
    # Note: V indices in strassen_steps above are relative to the block start?
    # No, I hardcoded 0, 1, 13, 14. I need to offset them.
    # Let's re-structure to generate dynamically.
    
    def add_strassen_block(base_v_col):
        nonlocal col_idx
        # base_v_col is 2*b
        v_b00 = base_v_col
        v_b01 = base_v_col + 1
        v_b10 = base_v_col + 13
        v_b11 = base_v_col + 14
        
        w_c00 = 4 * (base_v_col // 2)
        w_c01 = 4 * (base_v_col // 2) + 2
        w_c10 = 4 * (base_v_col // 2) + 1
        w_c11 = 4 * (base_v_col // 2) + 3
        
        # M1
        U[0][col_idx] += 1
        U[3][col_idx] += 1
        V[v_b00][col_idx] += 1
        V[v_b11][col_idx] += 1
        W[w_c00][col_idx] += 1
        W[w_c11][col_idx] += 1
        col_idx += 1
        
        # M2
        U[2][col_idx] += 1
        U[3][col_idx] += 1
        V[v_b00][col_idx] += 1
        W[w_c10][col_idx] += 1
        W[w_c11][col_idx] -= 1
        col_idx += 1
        
        # M3
        U[0][col_idx] += 1
        V[v_b10][col_idx] += 1
        V[v_b11][col_idx] -= 1
        W[w_c01][col_idx] += 1
        W[w_c11][col_idx] += 1
        col_idx += 1
        
        # M4
        U[3][col_idx] += 1
        V[v_b01][col_idx] += 1
        V[v_b00][col_idx] -= 1
        W[w_c00][col_idx] += 1
        W[w_c10][col_idx] += 1
        col_idx += 1
        
        # M5
        U[0][col_idx] += 1
        U[1][col_idx] += 1
        V[v_b11][col_idx] += 1
        W[w_c00][col_idx] -= 1
        W[w_c01][col_idx] += 1
        col_idx += 1
        
        # M6
        U[2][col_idx] += 1
        U[0][col_idx] -= 1
        V[v_b00][col_idx] += 1
        V[v_b10][col_idx] += 1
        W[w_c11][col_idx] += 1
        col_idx += 1
        
        # M7
        U[1][col_idx] += 1
        U[3][col_idx] -= 1
        V[v_b11][col_idx] += 1
        V[v_b01][col_idx] += 1
        W[w_c00][col_idx] += 1
        col_idx += 1

    for b in range(6):
        add_strassen_block(2 * b)
        
    # Last block: 2x2x1 (Column 12)
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Rank 4 decomposition:
    # M1 = A00 B0 -> C0
    # M2 = A01 B1 -> C0
    # M3 = A10 B0 -> C1
    # M4 = A11 B1 -> C1
    
    v_b0 = 12
    v_b1 = 25
    w_c0 = 24
    w_c1 = 25
    
    # M1
    U[0][col_idx] += 1
    V[v_b0][col_idx] += 1
    W[w_c0][col_idx] += 1
    col_idx += 1
    
    # M2
    U[1][col_idx] += 1
    V[v_b1][col_idx] += 1
    W[w_c0][col_idx] += 1
    col_idx += 1
    
    # M3
    U[2][col_idx] += 1
    V[v_b0][col_idx] += 1
    W[w_c1][col_idx] += 1
    col_idx += 1
    
    # M4
    U[3][col_idx] += 1
    V[v_b1][col_idx] += 1
    W[w_c1][col_idx] += 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
