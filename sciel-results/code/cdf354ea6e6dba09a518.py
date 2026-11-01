"""Matrix multiplication 2x2x11 with Strassen-based optimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # Strassen products for 2x2 matrix multiplication
    # Each: (U_dict for A, V_dict for B_pair, W_dict for C_pair)
    # A indices: (i,j) for i,j in {0,1}
    # B_pair indices: (j, k_local) for j in {0,1}, k_local in {0,1} (0=k1, 1=k2)
    # C_pair indices: (k_local, i) for k_local in {0,1}, i in {0,1}
    
    strassen_products = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    U_rows = n * m  # 4
    V_rows = m * p  # 22
    W_rows = p * n  # 22
    
    columns_U = []
    columns_V = []
    columns_W = []
    
    def add_strassen_pair(k1, k2):
        for (u_dict, v_dict, w_dict) in strassen_products:
            U_col = [0.0] * U_rows
            V_col = [0.0] * V_rows
            W_col = [0.0] * W_rows
            
            for (i, j), val in u_dict.items():
                U_col[i * m + j] = val
            
            for (j, k_local), val in v_dict.items():
                k_global = k1 if k_local == 0 else k2
                V_col[j * p + k_global] = val
            
            for (k_local, i), val in w_dict.items():
                k_global = k1 if k_local == 0 else k2
                W_col[k_global * n + i] = val
            
            columns_U.append(U_col)
            columns_V.append(V_col)
            columns_W.append(W_col)
    
    def add_schoolbook_single(k):
        for j in range(m):
            U_col = [0.0] * U_rows
            V_col = [0.0] * V_rows
            W_col = [0.0] * W_rows
            
            for i in range(n):
                U_col[i * m + j] = 1.0
            V_col[j * p + k] = 1.0
            for i in range(n):
                W_col[k * n + i] = 1.0
            
            columns_U.append(U_col)
            columns_V.append(V_col)
            columns_W.append(W_col)
    
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    singles = [10]
    
    for k1, k2 in pairs:
        add_strassen_pair(k1, k2)
    
    for k in singles:
        add_schoolbook_single(k)
    
    rank = len(columns_U)
    
    # Convert to (rows, rank) format
    U = [[columns_U[c][r] for c in range(rank)] for r in range(U_rows)]
    V = [[columns_V[c][r] for c in range(rank)] for r in range(V_rows)]
    W = [[columns_W[c][r] for c in range(rank)] for r in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
