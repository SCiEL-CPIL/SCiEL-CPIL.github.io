import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)
    
    # We will build lists of entries for U, V, W
    # Each entry is (row, col, value)
    U_entries = []
    V_entries = []
    W_entries = []
    col = 0
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1
        U_entries.append((i0*m + j0, col, 1))
        U_entries.append((i1*m + j1, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        V_entries.append((j1*p + k1, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        W_entries.append((k1*n + i1, col, 1))
        col += 1
        
        # M2
        U_entries.append((i1*m + j0, col, 1))
        U_entries.append((i1*m + j1, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        W_entries.append((k0*n + i1, col, 1))
        col += 1
        
        # M3
        U_entries.append((i0*m + j0, col, 1))
        V_entries.append((j1*p + k0, col, 1))
        V_entries.append((j1*p + k1, col, -1))
        W_entries.append((k1*n + i0, col, 1))
        col += 1
        
        # M4
        U_entries.append((i1*m + j1, col, 1))
        V_entries.append((j0*p + k1, col, 1))
        V_entries.append((j0*p + k0, col, -1))
        W_entries.append((k0*n + i0, col, 1))
        W_entries.append((k1*n + i1, col, 1))
        col += 1
        
        # M5
        U_entries.append((i0*m + j0, col, 1))
        U_entries.append((i0*m + j1, col, 1))
        V_entries.append((j1*p + k1, col, 1))
        W_entries.append((k0*n + i0, col, -1))
        W_entries.append((k1*n + i0, col, 1))
        col += 1
        
        # M6
        U_entries.append((i1*m + j0, col, 1))
        U_entries.append((i0*m + j0, col, -1))
        V_entries.append((j0*p + k0, col, 1))
        V_entries.append((j0*p + k1, col, 1))
        W_entries.append((k1*n + i1, col, 1))
        col += 1
        
        # M7
        U_entries.append((i0*m + j1, col, 1))
        U_entries.append((i1*m + j1, col, -1))
        V_entries.append((j1*p + k0, col, 1))
        V_entries.append((j1*p + k1, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1

    def add_schoolbook_221(i0, i1, j0, j1, k0):
        nonlocal col
        # C[i0, k0] += A[i0, j0] B[j0, k0]
        U_entries.append((i0*m + j0, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1
        
        # C[i0, k0] += A[i0, j1] B[j1, k0]
        U_entries.append((i0*m + j1, col, 1))
        V_entries.append((j1*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1
        
        # C[i1, k0] += A[i1, j0] B[j0, k0]
        U_entries.append((i1*m + j0, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        W_entries.append((k0*n + i1, col, 1))
        col += 1
        
        # C[i1, k0] += A[i1, j1] B[j1, k0]
        U_entries.append((i1*m + j1, col, 1))
        V_entries.append((j1*p + k0, col, 1))
        W_entries.append((k0*n + i1, col, 1))
        col += 1

    def add_schoolbook_122(i0, j0, j1, k0, k1):
        nonlocal col
        # C[i0, k0] += A[i0, j0] B[j0, k0]
        U_entries.append((i0*m + j0, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1
        
        # C[i0, k0] += A[i0, j1] B[j1, k0]
        U_entries.append((i0*m + j1, col, 1))
        V_entries.append((j1*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1
        
        # C[i0, k1] += A[i0, j0] B[j0, k1]
        U_entries.append((i0*m + j0, col, 1))
        V_entries.append((j0*p + k1, col, 1))
        W_entries.append((k1*n + i0, col, 1))
        col += 1
        
        # C[i0, k1] += A[i0, j1] B[j1, k1]
        U_entries.append((i0*m + j1, col, 1))
        V_entries.append((j1*p + k1, col, 1))
        W_entries.append((k1*n + i0, col, 1))
        col += 1

    def add_schoolbook_121(i0, j0, j1, k0):
        nonlocal col
        # C[i0, k0] += A[i0, j0] B[j0, k0]
        U_entries.append((i0*m + j0, col, 1))
        V_entries.append((j0*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1
        
        # C[i0, k0] += A[i0, j1] B[j1, k0]
        U_entries.append((i0*m + j1, col, 1))
        V_entries.append((j1*p + k0, col, 1))
        W_entries.append((k0*n + i0, col, 1))
        col += 1

    # Block 1: i in {0,1}, j in {0,1}
    # k in {0,1}, {2,3}, {4,5}, {6}
    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 0, 1, 4, 5)
    add_schoolbook_221(0, 1, 0, 1, 6)
    
    # Block 2: i in {0,1}, j in {2,3}
    # k in {0,1}, {2,3}, {4,5}, {6}
    add_strassen(0, 1, 2, 3, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3)
    add_strassen(0, 1, 2, 3, 4, 5)
    add_schoolbook_221(0, 1, 2, 3, 6)
    
    # Block 3: i in {2}, j in {0,1}
    # k in {0,1}, {2,3}, {4,5}, {6}
    add_schoolbook_122(2, 0, 1, 0, 1)
    add_schoolbook_122(2, 0, 1, 2, 3)
    add_schoolbook_122(2, 0, 1, 4, 5)
    add_schoolbook_121(2, 0, 1, 6)
    
    # Block 4: i in {2}, j in {2,3}
    # k in {0,1}, {2,3}, {4,5}, {6}
    add_schoolbook_122(2, 2, 3, 0, 1)
    add_schoolbook_122(2, 2, 3, 2, 3)
    add_schoolbook_122(2, 2, 3, 4, 5)
    add_schoolbook_121(2, 2, 3, 6)
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, c, val in U_entries:
        U[r][c] += val
    for r, c, val in V_entries:
        V[r][c] += val
    for r, c, val in W_entries:
        W[r][c] += val
        
    return {"U": U, "V": V, "W": W}
