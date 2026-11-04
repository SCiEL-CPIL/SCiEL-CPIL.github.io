import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # We will build the matrices dynamically
    U = []
    V = []
    W = []
    
    # Helper to add a column
    def add_col(u_vals, v_vals, w_vals):
        # u_vals: dict {row_index: val}
        # v_vals: dict {row_index: val}
        # w_vals: dict {row_index: val}
        U.append([0] * (n*m) if False else [0.0] * (n*m)) # Placeholder, we build lists
        # Actually, let's build lists of columns first
        pass

    # Let's use lists of columns
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_term(u_row, v_row, w_row, coeff=1):
        cols_U.append({u_row: coeff})
        cols_V.append({v_row: coeff})
        cols_W.append({w_row: coeff})
        
    # This is inefficient for Strassen which has multiple non-zeros.
    # Let's store columns as dicts.
    
    columns = []
    
    def add_column(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
        
    # Schoolbook block
    def add_schoolbook(i_start, j_start, k_start, i_size, j_size, k_size):
        for u in range(i_size):
            for v in range(j_size):
                for w in range(k_size):
                    i_global = i_start + u
                    j_global = j_start + v
                    k_global = k_start + w
                    
                    u_row = i_global * m + j_global
                    v_row = j_global * p + k_global
                    w_row = k_global * n + i_global
                    
                    add_column({u_row: 1}, {v_row: 1}, {w_row: 1})
                    
    # Strassen block (2x2x2)
    def add_strassen(i_start, j_start, k_start):
        # i_start, j_start, k_start are the global start indices for the 2x2 block
        # Local indices 0,1 map to i_start, i_start+1 etc.
        
        # M1: (A00+A11)(B00+B11)
        u = { (i_start)*m + j_start: 1, (i_start+1)*m + (j_start+1): 1 }
        v = { j_start*p + k_start: 1, (j_start+1)*p + (k_start+1): 1 }
        w = { k_start*n + i_start: 1 } # C00
        add_column(u, v, w)
        
        # M2: (A10+A11)B00
        u = { (i_start+1)*m + j_start: 1, (i_start+1)*m + (j_start+1): 1 }
        v = { j_start*p + k_start: 1 }
        w = { k_start*n + (i_start+1): 1 } # C10
        add_column(u, v, w)
        
        # M3: A00(B10-B11)
        u = { i_start*m + j_start: 1 }
        v = { (j_start+1)*p + k_start: 1, (j_start+1)*p + (k_start+1): -1 }
        w = { (k_start+1)*n + i_start: 1, (k_start+1)*n + (i_start+1): 1 } # C01, C11
        add_column(u, v, w)
        
        # M4: A11(B01-B10)
        u = { (i_start+1)*m + (j_start+1): 1 }
        v = { j_start*p + (k_start+1): 1, (j_start+1)*p + k_start: -1 }
        w = { k_start*n + i_start: 1, k_start*n + (i_start+1): 1 } # C00, C10
        add_column(u, v, w)
        
        # M5: (A00+A01)B11
        u = { i_start*m + j_start: 1, i_start*m + (j_start+1): 1 }
        v = { (j_start+1)*p + (k_start+1): 1 }
        w = { k_start*n + i_start: -1, (k_start+1)*n + i_start: 1 } # C00, C01
        add_column(u, v, w)
        
        # M6: (A10-A00)(B00+B01)
        u = { (i_start+1)*m + j_start: 1, i_start*m + j_start: -1 }
        v = { j_start*p + k_start: 1, j_start*p + (k_start+1): 1 }
        w = { (k_start+1)*n + (i_start+1): 1 } # C11
        add_column(u, v, w)
        
        # M7: (A01-A11)(B10+B11)
        u = { i_start*m + (j_start+1): 1, (i_start+1)*m + (j_start+1): -1 }
        v = { (j_start+1)*p + k_start: 1, (j_start+1)*p + (k_start+1): 1 }
        w = { k_start*n + i_start: 1 } # C00
        add_column(u, v, w)
        
    # Block definitions
    # A: 4x5. Rows 0-1 (I1), 2-3 (I2). Cols 0-1 (J1), 2-3 (J2), 4 (J3).
    # B: 5x5. Rows 0-1 (J1), 2-3 (J2), 4 (J3). Cols 0-1 (K1), 2-3 (K2), 4 (K3).
    
    I_blocks = [(0, 2), (2, 2)] # start, size
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 1)]
    
    for i_idx, (i_start, i_size) in enumerate(I_blocks):
        for j_idx, (j_start, j_size) in enumerate(J_blocks):
            for k_idx, (k_start, k_size) in enumerate(K_blocks):
                if j_size == 2 and k_size == 2 and i_size == 2:
                    add_strassen(i_start, j_start, k_start)
                else:
                    add_schoolbook(i_start, j_start, k_start, i_size, j_size, k_size)
                    
    rank = len(columns)
    
    # Convert to matrices
    U_mat = [[0] * rank for _ in range(n*m)]
    V_mat = [[0] * rank for _ in range(m*p)]
    W_mat = [[0] * rank for _ in range(p*n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(columns):
        for row, val in u_dict.items():
            U_mat[row][c] = val
        for row, val in v_dict.items():
            V_mat[row][c] = val
        for row, val in w_dict.items():
            W_mat[row][c] = val
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
