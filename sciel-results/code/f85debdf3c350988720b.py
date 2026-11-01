import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank)
    
    # Rank calculation
    # 6 Strassen blocks (2x2x2) -> 6 * 7 = 42
    # 3 Schoolbook blocks (2x1x2) -> 3 * 4 = 12
    # Total rank = 54
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(u_col_start, v_row_start, v_col_start, w_row_start):
        nonlocal col
        # u_col_start: start col of A (0 or 2)
        # v_row_start: start row of B (0 or 2)
        # v_col_start: start col of B (0, 2, or 4)
        # w_row_start: start col of C (0, 2, or 4)
        
        # A indices: (0, u_col_start), (0, u_col_start+1), (1, u_col_start), (1, u_col_start+1)
        # B indices: (v_row_start, v_col_start), (v_row_start, v_col_start+1), (v_row_start+1, v_col_start), (v_row_start+1, v_col_start+1)
        # C indices: (w_row_start, 0), (w_row_start, 1), (w_row_start+1, 0), (w_row_start+1, 1)
        
        # M1: (a+d)(e+h)
        # U: A[0, u_col_start] + A[1, u_col_start+1]
        # V: B[v_row_start, v_col_start] + B[v_row_start+1, v_col_start+1]
        # W: C[w_row_start, 0] + C[w_row_start+1, 1]
        
        U[0 * m + u_col_start][col] = 1.0
        U[1 * m + u_col_start + 1][col] = 1.0
        V[v_row_start * p + v_col_start][col] = 1.0
        V[(v_row_start + 1) * p + v_col_start + 1][col] = 1.0
        W[w_row_start * n + 0][col] = 1.0
        W[(w_row_start + 1) * n + 1][col] = 1.0
        col += 1
        
        # M2: (c+d)e
        # U: A[1, u_col_start] + A[1, u_col_start+1]
        # V: B[v_row_start, v_col_start]
        # W: C[w_row_start+1, 0] - C[w_row_start+1, 1]
        
        U[1 * m + u_col_start][col] = 1.0
        U[1 * m + u_col_start + 1][col] = 1.0
        V[v_row_start * p + v_col_start][col] = 1.0
        W[(w_row_start + 1) * n + 0][col] = 1.0
        W[(w_row_start + 1) * n + 1][col] = -1.0
        col += 1
        
        # M3: a(f-h)
        # U: A[0, u_col_start]
        # V: B[v_row_start, v_col_start+1] - B[v_row_start+1, v_col_start+1]
        # W: C[w_row_start, 1] + C[w_row_start+1, 1]
        
        U[0 * m + u_col_start][col] = 1.0
        V[v_row_start * p + v_col_start + 1][col] = 1.0
        V[(v_row_start + 1) * p + v_col_start + 1][col] = -1.0
        W[w_row_start * n + 1][col] = 1.0
        W[(w_row_start + 1) * n + 1][col] = 1.0
        col += 1
        
        # M4: d(g-e)
        # U: A[1, u_col_start+1]
        # V: B[v_row_start+1, v_col_start] - B[v_row_start, v_col_start]
        # W: C[w_row_start, 0] + C[w_row_start+1, 0]
        
        U[1 * m + u_col_start + 1][col] = 1.0
        V[(v_row_start + 1) * p + v_col_start][col] = 1.0
        V[v_row_start * p + v_col_start][col] = -1.0
        W[w_row_start * n + 0][col] = 1.0
        W[(w_row_start + 1) * n + 0][col] = 1.0
        col += 1
        
        # M5: (a+b)h
        # U: A[0, u_col_start] + A[0, u_col_start+1]
        # V: B[v_row_start+1, v_col_start+1]
        # W: -C[w_row_start, 0] + C[w_row_start, 1]
        
        U[0 * m + u_col_start][col] = 1.0
        U[0 * m + u_col_start + 1][col] = 1.0
        V[(v_row_start + 1) * p + v_col_start + 1][col] = 1.0
        W[w_row_start * n + 0][col] = -1.0
        W[w_row_start * n + 1][col] = 1.0
        col += 1
        
        # M6: (c-a)(e+f)
        # U: A[1, u_col_start] - A[0, u_col_start]
        # V: B[v_row_start, v_col_start] + B[v_row_start, v_col_start+1]
        # W: C[w_row_start+1, 1]
        
        U[1 * m + u_col_start][col] = 1.0
        U[0 * m + u_col_start][col] = -1.0
        V[v_row_start * p + v_col_start][col] = 1.0
        V[v_row_start * p + v_col_start + 1][col] = 1.0
        W[(w_row_start + 1) * n + 1][col] = 1.0
        col += 1
        
        # M7: (b-d)(g+h)
        # U: A[0, u_col_start+1] - A[1, u_col_start+1]
        # V: B[v_row_start+1, v_col_start] + B[v_row_start+1, v_col_start+1]
        # W: C[w_row_start, 0]
        
        U[0 * m + u_col_start + 1][col] = 1.0
        U[1 * m + u_col_start + 1][col] = -1.0
        V[(v_row_start + 1) * p + v_col_start][col] = 1.0
        V[(v_row_start + 1) * p + v_col_start + 1][col] = 1.0
        W[w_row_start * n + 0][col] = 1.0
        col += 1

    # Helper to add 2x1x2 block
    def add_212(u_col, v_row, v_col_start, w_row_start):
        nonlocal col
        # A: (0, u_col), (1, u_col)
        # B: (v_row, v_col_start), (v_row, v_col_start+1)
        # C: (w_row_start, 0), (w_row_start, 1), (w_row_start+1, 0), (w_row_start+1, 1)
        
        # T1: A[0, u_col] * B[v_row, v_col_start] -> C[w_row_start, 0]
        U[0 * m + u_col][col] = 1.0
        V[v_row * p + v_col_start][col] = 1.0
        W[w_row_start * n + 0][col] = 1.0
        col += 1
        
        # T2: A[0, u_col] * B[v_row, v_col_start+1] -> C[w_row_start, 1]
        U[0 * m + u_col][col] = 1.0
        V[v_row * p + v_col_start + 1][col] = 1.0
        W[w_row_start * n + 1][col] = 1.0
        col += 1
        
        # T3: A[1, u_col] * B[v_row, v_col_start] -> C[w_row_start+1, 0]
        U[1 * m + u_col][col] = 1.0
        V[v_row * p + v_col_start][col] = 1.0
        W[(w_row_start + 1) * n + 0][col] = 1.0
        col += 1
        
        # T4: A[1, u_col] * B[v_row, v_col_start+1] -> C[w_row_start+1, 1]
        U[1 * m + u_col][col] = 1.0
        V[v_row * p + v_col_start + 1][col] = 1.0
        W[(w_row_start + 1) * n + 1][col] = 1.0
        col += 1

    # Apply Strassen blocks
    # A cols 0-1, B rows 0-1
    add_strassen(0, 0, 0, 0)
    add_strassen(0, 0, 2, 2)
    add_strassen(0, 0, 4, 4)
    # A cols 2-3, B rows 2-3
    add_strassen(2, 2, 0, 0)
    add_strassen(2, 2, 2, 2)
    add_strassen(2, 2, 4, 4)
    
    # Apply 2x1x2 blocks
    # A col 4, B row 4
    add_212(4, 4, 0, 0)
    add_212(4, 4, 2, 2)
    add_212(4, 4, 4, 4)
    
    return {"U": U, "V": V, "W": W}
