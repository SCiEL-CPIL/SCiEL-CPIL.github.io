import time
import numpy as np

def construct():
    n, m, p = 2, 5, 5
    total_rank = 46
    
    U = [[0] * total_rank for _ in range(n * m)]  # 10 x 46
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 46
    W = [[0] * total_rank for _ in range(p * n)]  # 10 x 46
    
    col = 0
    
    def add_strassen_2x2_2x2(a_cols, b_rows, c_cols):
        """Strassen for A[:,a_cols] (2x2) * B[b_rows,:][:,c_cols] (2x2) -> C[:,c_cols] (2x2)"""
        nonlocal col
        a0, a1 = a_cols
        b0, b1 = b_rows
        c0, c1 = c_cols
        
        # M1 = (A[0,a0]+A[1,a1])*(B[b0,c0]+B[b1,c1])
        U[0*5+a0][col] = 1; U[1*5+a1][col] = 1
        V[b0*5+c0][col] = 1; V[b1*5+c1][col] = 1
        W[c0*2+0][col] = 1; W[c1*2+1][col] = 1
        col += 1
        
        # M2 = (A[1,a0]+A[1,a1])*B[b0,c0]
        U[1*5+a0][col] = 1; U[1*5+a1][col] = 1
        V[b0*5+c0][col] = 1
        W[c0*2+1][col] = 1
        col += 1
        
        # M3 = A[0,a0]*(B[b0,c1]-B[b1,c1])
        U[0*5+a0][col] = 1
        V[b0*5+c1][col] = 1; V[b1*5+c1][col] = -1
        W[c1*2+0][col] = 1
        col += 1
        
        # M4 = A[1,a1]*(B[b1,c0]-B[b0,c0])
        U[1*5+a1][col] = 1
        V[b1*5+c0][col] = 1; V[b0*5+c0][col] = -1
        W[c0*2+0][col] = 1; W[c0*2+1][col] = 1
        col += 1
        
        # M5 = (A[0,a0]+A[0,a1])*B[b1,c1]
        U[0*5+a0][col] = 1; U[0*5+a1][col] = 1
        V[b1*5+c1][col] = 1
        W[c0*2+0][col] = -1; W[c1*2+0][col] = 1
        col += 1
        
        # M6 = (A[1,a0]-A[0,a0])*(B[b0,c0]+B[b0,c1])
        U[1*5+a0][col] = 1; U[0*5+a0][col] = -1
        V[b0*5+c0][col] = 1; V[b0*5+c1][col] = 1
        W[c1*2+1][col] = 1
        col += 1
        
        # M7 = (A[0,a1]-A[1,a1])*(B[b1,c0]+B[b1,c1])
        U[0*5+a1][col] = 1; U[1*5+a1][col] = -1
        V[b1*5+c0][col] = 1; V[b1*5+c1][col] = 1
        W[c0*2+0][col] = 1; W[c1*2+1][col] = 1
        col += 1
    
    def add_naive_2x2_2x1(a_cols, b_row, c_col):
        """Naive for A[:,a_cols] (2x2) * B[b_row:b_row+2, c_col] (2x1) -> C[:,c_col] (2x1)"""
        nonlocal col
        a0, a1 = a_cols
        # 4 terms
        for i in range(2):
            for j_idx, a_j in enumerate(a_cols):
                b_j = b_row + j_idx
                U[i*5+a_j][col] = 1
                V[b_j*5+c_col][col] = 1
                W[c_col*2+i][col] = 1
                col += 1
    
    # Block A₁ (cols 0,1) × Block B₁ (rows 0,1)
    add_strassen_2x2_2x2((0, 1), (0, 1), (0, 1))
    add_strassen_2x2_2x2((0, 1), (0, 1), (2, 3))
    add_naive_2x2_2x1((0, 1), 0, 4)
    
    # Block A₂ (cols 2,3) × Block B₂ (rows 2,3)
    add_strassen_2x2_2x2((2, 3), (2, 3), (0, 1))
    add_strassen_2x2_2x2((2, 3), (2, 3), (2, 3))
    add_naive_2x2_2x1((2, 3), 2, 4)
    
    # Block A₃ (col 4) × Block B₃ (row 4)
    for k in range(5):
        U[0*5+4][col] = 1; V[4*5+k][col] = 1; W[k*2+0][col] = 1; col += 1
        U[1*5+4][col] = 1; V[4*5+k][col] = 1; W[k*2+1][col] = 1; col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    return {"U": U, "V": V, "W": W}
