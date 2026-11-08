import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 2, 4, 5
    
    # Max rank estimate: 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def add_strassen_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # A block: rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        # B block: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        # C block: rows a_row_offset..a_row_offset+1, cols b_col_offset..b_col_offset+1
        
        # Strassen M1 = A00 * (B00 + B11)
        U[u_idx(a_row_offset+0, a_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+0)][col] += 1
        W[w_idx(b_col_offset+1, a_row_offset+1)][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[u_idx(a_row_offset+1, a_col_offset+0)][col] += 1
        U[u_idx(a_row_offset+1, a_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+0)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+1)][col] += 1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[u_idx(a_row_offset+0, a_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] -= 1
        W[w_idx(b_col_offset+1, a_row_offset+0)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+0)][col] -= 1 # Wait, M3 contributes to C01 and C11?
        # C01 = M3 + M5
        # C11 = M1 - M3 + M2 - M6
        # So M3 adds to C01 and subtracts from C11.
        W[w_idx(b_col_offset+1, a_row_offset+0)][col] += 1
        W[w_idx(b_col_offset+1, a_row_offset+1)][col] -= 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[u_idx(a_row_offset+1, a_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+0)][col] -= 1
        W[w_idx(b_col_offset+1, a_row_offset+0)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+1)][col] += 1
        col += 1
        
        # M5 = (A00 + A11) * (B11 + B00)
        U[u_idx(a_row_offset+0, a_col_offset+0)][col] += 1
        U[u_idx(a_row_offset+1, a_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+0)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+0)][col] -= 1
        W[w_idx(b_col_offset+1, a_row_offset+0)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+1)][col] += 1
        W[w_idx(b_col_offset+1, a_row_offset+1)][col] -= 1
        col += 1
        
        # M6 = (A01 - A11) * (B10 + B11)
        U[u_idx(a_row_offset+0, a_col_offset+1)][col] += 1
        U[u_idx(a_row_offset+1, a_col_offset+1)][col] -= 1
        V[v_idx(b_row_offset+1, b_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset+1)][col] += 1
        W[w_idx(b_col_offset+1, a_row_offset+1)][col] -= 1
        col += 1
        
        # M7 = (A10 - A00) * (B00 + B01)
        U[u_idx(a_row_offset+1, a_col_offset+0)][col] += 1
        U[u_idx(a_row_offset+0, a_col_offset+0)][col] -= 1
        V[v_idx(b_row_offset+0, b_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset+1)][col] += 1
        W[w_idx(b_col_offset+0, a_row_offset+0)][col] += 1
        col += 1

    def add_standard_2x2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        nonlocal col
        # A block: 2x2, B block: 2x1, C block: 2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # T1: A00 B00 -> C00
        U[u_idx(a_row_offset+0, a_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset)][col] += 1
        W[w_idx(b_col_offset, a_row_offset+0)][col] += 1
        col += 1
        
        # T2: A01 B10 -> C00
        U[u_idx(a_row_offset+0, a_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset)][col] += 1
        W[w_idx(b_col_offset, a_row_offset+0)][col] += 1
        col += 1
        
        # T3: A10 B00 -> C10
        U[u_idx(a_row_offset+1, a_col_offset+0)][col] += 1
        V[v_idx(b_row_offset+0, b_col_offset)][col] += 1
        W[w_idx(b_col_offset, a_row_offset+1)][col] += 1
        col += 1
        
        # T4: A11 B10 -> C10
        U[u_idx(a_row_offset+1, a_col_offset+1)][col] += 1
        V[v_idx(b_row_offset+1, b_col_offset)][col] += 1
        W[w_idx(b_col_offset, a_row_offset+1)][col] += 1
        col += 1

    # A1: rows 0-1, cols 0-1. B1: rows 0-1, cols 0-4.
    # Split B1 cols into 0-1, 2-3, 4.
    add_strassen_2x2x2(0, 0, 0, 0) # C cols 0,1
    add_strassen_2x2x2(0, 0, 0, 2) # C cols 2,3
    add_standard_2x2x1(0, 0, 0, 4) # C col 4
    
    # A2: rows 0-1, cols 2-3. B2: rows 2-3, cols 0-4.
    # Split B2 cols into 0-1, 2-3, 4.
    add_strassen_2x2x2(0, 2, 2, 0) # C cols 0,1
    add_strassen_2x2x2(0, 2, 2, 2) # C cols 2,3
    add_standard_2x2x1(0, 2, 2, 4) # C col 4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
