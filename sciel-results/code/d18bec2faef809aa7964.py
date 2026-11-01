import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Total rank expected: 60.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: dict {(i,j): val}
        v_vals: dict {(j,k): val}
        w_vals: dict {(k,i): val}
        """
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Helper to add Strassen decomposition for a 2x2x2 block
    # i in [i0, i0+1], j in [j0, j0+1], k in [k0, k0+1]
    def add_strassen(i0, j0, k0):
        # Local indices 0,1 map to global i0, i0+1 etc.
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11, C01, C10 coeffs
        
        # We need to define the 7 terms.
        # Term 1: M1
        u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        # C00 += M1, C11 += M1, C01 -= M7? No, W is the output coeff.
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M4
        # C10 = M2 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # We add 7 columns.
        
        # Col 1 (M1)
        w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
        add_term(u1, v1, w1)
        
        # Col 2 (M2)
        u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0, i0+1): 1, (k0+1, i0+1): -1} # C10 += M2, C11 -= M2
        add_term(u2, v2, w2)
        
        # Col 3 (M3)
        u3 = {(i0, j0): 1}
        v3 = {(j0+1, k0): -1, (j0+1, k0+1): 1} # B10 - B11
        w3 = {(k0, i0): 1, (k0+1, i0+1): 1} # C01 += M3, C11 += M3
        add_term(u3, v3, w3)
        
        # Col 4 (M4)
        u4 = {(i0+1, j0+1): 1}
        v4 = {(j0, k0+1): 1, (j0, k0): -1} # B01 - B00
        w4 = {(k0, i0): -1, (k0+1, i0): 1} # C00 -= M4, C01 += M4
        add_term(u4, v4, w4)
        
        # Col 5 (M5)
        u5 = {(i0, j0): 1, (i0, j0+1): 1}
        v5 = {(j0+1, k0+1): 1}
        w5 = {(k0, i0): 1} # C00 += M5
        add_term(u5, v5, w5)
        
        # Col 6 (M6)
        u6 = {(i0+1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k0+1): 1}
        w6 = {(k0+1, i0+1): 1} # C11 += M6
        add_term(u6, v6, w6)
        
        # Col 7 (M7)
        u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w7 = {(k0, i0): -1, (k0, i0+1): 1} # C00 -= M7, C10 += M7
        add_term(u7, v7, w7)

    # Block 1: k=0,1
    add_strassen(0, 0, 0)
    # Cross terms for Block 1 (k=0,1)
    # C_0:2, 0:2 involves A_0:2,2 and B_2,0:2
    # A_12 (2x1) * B_21 (1x2)
    # i in 0,1; j=2; k in 0,1
    # C_ik = A_i2 B_2k
    # Rank 4 terms
    for i in [0, 1]:
        for k in [0, 1]:
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)
            
    # C_2, 0:2 involves A_2,0:2 and B_0:2,0:2 AND A_2,2 and B_2,0:2
    # A_21 (1x2) * B_11 (2x2)
    # i=2; j in 0,1; k in 0,1
    # C_2k = sum_j A_2j B_jk
    for j in [0, 1]:
        for k in [0, 1]:
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(k, 2): 1}
            add_term(u, v, w)
            
    # A_22 (1x1) * B_21 (1x2)
    # i=2; j=2; k in 0,1
    for k in [0, 1]:
        u = {(2, 2): 1}
        v = {(2, k): 1}
        w = {(k, 2): 1}
        add_term(u, v, w)

    # Block 2: k=2,3
    add_strassen(0, 0, 2)
    # Cross terms for Block 2
    for i in [0, 1]:
        for k in [2, 3]:
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)
    for j in [0, 1]:
        for k in [2, 3]:
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(k, 2): 1}
            add_term(u, v, w)
    for k in [2, 3]:
        u = {(2, 2): 1}
        v = {(2, k): 1}
        w = {(k, 2): 1}
        add_term(u, v, w)

    # Block 3: k=4,5
    add_strassen(0, 0, 4)
    # Cross terms for Block 3
    for i in [0, 1]:
        for k in [4, 5]:
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)
    for j in [0, 1]:
        for k in [4, 5]:
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(k, 2): 1}
            add_term(u, v, w)
    for k in [4, 5]:
        u = {(2, 2): 1}
        v = {(2, k): 1}
        w = {(k, 2): 1}
        add_term(u, v, w)

    # Block 4: k=6
    # Schoolbook for 3x3x1
    for i in range(3):
        for j in range(3):
            u = {(i, j): 1}
            v = {(j, 6): 1}
            w = {(6, i): 1}
            add_term(u, v, w)

    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

