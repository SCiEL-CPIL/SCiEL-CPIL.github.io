import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    U_rows = n * m  # 9
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    # We will build the factors as lists of (row, col, coeff) tuples, then convert to matrices
    # Or just initialize matrices with zeros and fill them.
    # Rank is 68.
    rank = 68
    
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        nonlocal col
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        W[w_idx][col] += w_coeff
        col += 1

    # 1. Strassen for A[0:2, 0:2] * B[0:2, 0:8]
    # Split B[0:2, 0:8] into 4 blocks of 2x2: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    for block_idx in range(4):
        # B block columns: 2*block_idx to 2*block_idx+2
        b_col_start = 2 * block_idx
        
        # Strassen terms
        # M1 = (A00 + A11)(B00 + B11)
        # A indices: A[0,0]=0, A[1,1]=4
        # B indices: B[0, b_col_start]=b_col_start, B[1, b_col_start+1]=8+b_col_start+1
        # W indices: C[0, b_col_start]=b_col_start*3+0, C[1, b_col_start+1]=b_col_start*3+1?
        # W index formula: k*n + i. k is col, i is row.
        # C[0, c] -> k=c, i=0 -> idx = c*3.
        # C[1, c] -> k=c, i=1 -> idx = c*3 + 1.
        
        # M1
        u_00 = 0
        u_11 = 4
        v_00 = b_col_start
        v_11 = 8 + b_col_start + 1
        w_00 = b_col_start * 3
        w_11 = (b_col_start + 1) * 3 + 1
        
        U[u_00][col] = 1.0
        U[u_11][col] = 1.0
        V[v_00][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = 1.0
        W[w_11][col] = 1.0
        col += 1
        
        # M2 = (A10 + A11) B00
        u_10 = 3
        v_00 = b_col_start
        w_10 = b_col_start * 3 + 1
        
        U[u_10][col] = 1.0
        U[u_11][col] = 1.0
        V[v_00][col] = 1.0
        W[w_10][col] = 1.0
        col += 1
        
        # M3 = A00 (B01 - B11)
        u_00 = 0
        v_01 = b_col_start + 1
        v_11 = 8 + b_col_start + 1
        w_01 = (b_col_start + 1) * 3
        
        U[u_00][col] = 1.0
        V[v_01][col] = 1.0
        V[v_11][col] = -1.0
        W[w_01][col] = 1.0
        W[w_11][col] = 1.0 # Wait, M3 contributes to C01 and C11?
        # Formula: C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M3 contributes to C01 (1) and C11 (1).
        col += 1
        
        # M4 = A11 (B10 - B00)
        u_11 = 4
        v_10 = 8 + b_col_start
        v_00 = b_col_start
        w_00 = b_col_start * 3
        w_10 = b_col_start * 3 + 1
        
        U[u_11][col] = 1.0
        V[v_10][col] = 1.0
        V[v_00][col] = -1.0
        W[w_00][col] = 1.0
        W[w_10][col] = 1.0
        col += 1
        
        # M5 = (A00 + A01) B11
        u_00 = 0
        u_01 = 1
        v_11 = 8 + b_col_start + 1
        w_00 = b_col_start * 3
        w_01 = (b_col_start + 1) * 3
        
        U[u_00][col] = 1.0
        U[u_01][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = -1.0
        W[w_01][col] = 1.0
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        u_10 = 3
        u_00 = 0
        v_00 = b_col_start
        v_01 = b_col_start + 1
        w_11 = (b_col_start + 1) * 3 + 1
        
        U[u_10][col] = 1.0
        U[u_00][col] = -1.0
        V[v_00][col] = 1.0
        V[v_01][col] = 1.0
        W[w_11][col] = 1.0
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        u_01 = 1
        u_11 = 4
        v_10 = 8 + b_col_start
        v_11 = 8 + b_col_start + 1
        w_00 = b_col_start * 3
        w_11 = (b_col_start + 1) * 3 + 1
        
        U[u_01][col] = 1.0
        U[u_11][col] = -1.0
        V[v_10][col] = 1.0
        V[v_11][col] = 1.0
        W[w_00][col] = 1.0
        W[w_11][col] = 1.0
        col += 1

    # 2. A[0:2, 2] * B[2, 0:8]
    # C[0, k] += A[0, 2] B[2, k]
    # C[1, k] += A[1, 2] B[2, k]
    for k in range(8):
        # Term for C[0, k]
        u_idx = 0 * 3 + 2  # A[0,2]
        v_idx = 2 * 8 + k  # B[2,k]
        w_idx = k * 3 + 0  # C[0,k]
        
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1
        
        # Term for C[1, k]
        u_idx = 1 * 3 + 2  # A[1,2]
        v_idx = 2 * 8 + k  # B[2,k]
        w_idx = k * 3 + 1  # C[1,k]
        
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1

    # 3. A[2, 0:2] * B[0:2, 0:8]
    # C[2, k] += A[2, 0] B[0, k] + A[2, 1] B[1, k]
    for k in range(8):
        # Term for A[2, 0] B[0, k]
        u_idx = 2 * 3 + 0  # A[2,0]
        v_idx = 0 * 8 + k  # B[0,k]
        w_idx = k * 3 + 2  # C[2,k]
        
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1
        
        # Term for A[2, 1] B[1, k]
        u_idx = 2 * 3 + 1  # A[2,1]
        v_idx = 1 * 8 + k  # B[1,k]
        w_idx = k * 3 + 2  # C[2,k]
        
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1

    # 4. A[2, 2] * B[2, 0:8]
    # C[2, k] += A[2, 2] B[2, k]
    for k in range(8):
        u_idx = 2 * 3 + 2  # A[2,2]
        v_idx = 2 * 8 + k  # B[2,k]
        w_idx = k * 3 + 2  # C[2,k]
        
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        W[w_idx][col] = 1.0
        col += 1

    return {"U": U, "V": V, "W": W}
