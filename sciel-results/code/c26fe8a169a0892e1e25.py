import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for a 2x2 block
    # a_indices: list of 4 global indices for A (00, 01, 10, 11)
    # b_indices: list of 4 global indices for B (00, 01, 10, 11)
    # c_indices: list of 4 global indices for C (00, 01, 10, 11)
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (a00 + a11) * (b00 + b11)
        # C00 += M1, C11 += M1
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        # C10 += M2, C11 -= M2
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        # C01 += M3, C11 += M3
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        # C00 += M4, C10 += M4
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        # C00 -= M5, C01 += M5
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        # C11 += M6
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        # C00 += M7
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1

    # Block 1: A1 B11 -> C11 (cols 0,1)
    # A1: i=0,1; j=0,1
    a1 = [0, 1, 4, 5]
    # B11: j=0,1; k=0,1
    b11 = [0, 1, 4, 5]
    # C11: k=0,1; i=0,1
    c11 = [0, 1, 2, 3]
    add_strassen(a1, b11, c11)
    
    # Block 2: A2 B21 -> C11 (cols 0,1)
    # A2: i=0,1; j=2,3
    a2 = [2, 3, 6, 7]
    # B21: j=2,3; k=0,1
    b21 = [8, 9, 12, 13]
    # C11: k=0,1; i=0,1
    c11 = [0, 1, 2, 3]
    add_strassen(a2, b21, c11)
    
    # Block 3: A1 B12 -> C21 (cols 2,3)
    # A1: i=0,1; j=0,1
    a1 = [0, 1, 4, 5]
    # B12: j=0,1; k=2,3
    b12 = [2, 3, 6, 7]
    # C21: k=2,3; i=0,1
    c21 = [4, 5, 6, 7]
    add_strassen(a1, b12, c21)
    
    # Block 4: A2 B22 -> C21 (cols 2,3)
    # A2: i=0,1; j=2,3
    a2 = [2, 3, 6, 7]
    # B22: j=2,3; k=2,3
    b22 = [10, 11, 14, 15]
    # C21: k=2,3; i=0,1
    c21 = [4, 5, 6, 7]
    add_strassen(a2, b22, c21)
    
    return {"U": U, "V": V, "W": W}
