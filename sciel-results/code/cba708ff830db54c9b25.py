import numpy as np

def construct():
    n, m, p = 5, 5, 5
    
    U_rows = []
    V_rows = []
    W_rows = []

    def add_term(u, v, w):
        U_rows.append(u)
        V_rows.append(v)
        W_rows.append(w)

    # 1. Terms for the top-left 4x4 block (indices 0-3)
    # Using Strassen's rank-19 decomposition
    # A11, A12, A21, A22 are 2x2 blocks of A (rows/cols 0-1, 2-3)
    # B11, B12, B21, B22 are 2x2 blocks of B (rows/cols 0-1, 2-3)
    
    # Helper to get indices
    def idx(r, c):
        return r * m + c
    
    # M1 = (A11 + A22) * (B11 + B22)
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i,j)] = 1
            u[idx(i+2,j+2)] = 1
            v[idx(j,i)] = 1
            v[idx(j+2,i+2)] = 1
            w[idx(i,i)] = 1
            w[idx(i+2,i+2)] = 1
            add_term(u, v, w)
    # M1 contributes to C11 and C22

    # M2 = (A21 + A22) * B11
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i+2,j)] = 1
            u[idx(i+2,j+2)] = 1
            v[idx(j,i)] = 1
            w[idx(i+2,i)] = 1
            add_term(u, v, w)
    # M2 contributes to C21 and C22 (with -1)

    # M3 = A11 * (B12 - B22)
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i,j)] = 1
            v[idx(j,i+2)] = 1
            v[idx(j+2,i)] = -1
            w[idx(i,i+2)] = 1
            add_term(u, v, w)
    # M3 contributes to C12 and C22

    # M4 = A22 * (B21 - B11)
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i+2,j+2)] = 1
            v[idx(j+2,i+2)] = 1
            v[idx(j,i)] = -1
            w[idx(i+2,i+2)] = 1
            add_term(u, v, w)
    # M4 contributes to C21 and C11

    # M5 = (A11 + A12) * B22
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i,j)] = 1
            u[idx(i,j+2)] = 1
            v[idx(j+2,i+2)] = 1
            w[idx(i,i+2)] = 1
            add_term(u, v, w)
    # M5 contributes to C12 and C11 (with -1)

    # M6 = (A21 - A11) * (B11 + B12)
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i+2,j)] = 1
            u[idx(i,j)] = -1
            v[idx(j,i)] = 1
            v[idx(j,i+2)] = 1
            w[idx(i+2,i)] = 1
            add_term(u, v, w)
    # M6 contributes to C22

    # M7 = (A12 - A22) * (B21 + B22)
    for i in range(2):
        for j in range(2):
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[idx(i,j+2)] = 1
            u[idx(i+2,j+2)] = -1
            v[idx(j+2,i+2)] = 1
            v[idx(j+2,i)] = 1
            w[idx(i+2,i+2)] = 1
            add_term(u, v, w)
    # M7 contributes to C11 and C22 (with -1)

    # 2. Terms for the 5th row and column (index 4)
    # We need to compute C[4, j] for j=0..4, C[i, 4] for i=0..4, and C[4,4]
    # Using schoolbook terms for these
    # C[i, 4] = sum_k A[i, k] B[k, 4]
    # C[4, j] = sum_k A[4, k] B[k, j]
    # C[4, 4] = sum_k A[4, k] B[k, 4]
    
    # Note: C[4,4] is counted once
    # C[i,4] for i=0..3: 4 terms
    # C[4,j] for j=0..3: 4 terms
    # C[4,4]: 1 term
    # Total 9 terms for the 5th row/col
    
    # Actually, we can use schoolbook for all terms involving index 4.
    # For i in 0..4, j in 0..4:
    #   If i==4 or j==4:
    #     Use schoolbook term A[i,k] * B[k,j] -> C[i,j]
    #     But we already handled i<4 and j<4 with Strassen.
    #     So we only need terms where i==4 or j==4.
    #     These are:
    #     - i=4, j=0..4: 5 terms
    #     - i=0..3, j=4: 4 terms
    #     Total 9 terms.
    
    # For each k=0..4:
    for k in range(5):
        # Term for C[4, k] from A[4, k] * B[k, 4]
        # Wait, C[4, j] = sum_k A[4, k] B[k, j]
        # So for fixed k, we have terms A[4, k] B[k, j] for all j.
        # But we can add them one by one.
        
        # Actually, let's just add schoolbook terms for all (i,j) where i==4 or j==4.
        # i=4, j=0
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(4, 0)] = 1
        v[idx(0, 4)] = 1
        w[idx(4, 4)] = 1
        add_term(u, v, w)
        
        # i=4, j=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(4, 1)] = 1
        v[idx(1, 4)] = 1
        w[idx(4, 4)] = 1
        add_term(u, v, w)
        
        # i=4, j=2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(4, 2)] = 1
        v[idx(2, 4)] = 1
        w[idx(4, 4)] = 1
        add_term(u, v, w)
        
        # i=4, j=3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(4, 3)] = 1
        v[idx(3, 4)] = 1
        w[idx(4, 4)] = 1
        add_term(u, v, w)
        
        # i=4, j=4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(4, 4)] = 1
        v[idx(4, 4)] = 1
        w[idx(4, 4)] = 1
        add_term(u, v, w)
        
        # i=0, j=4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(0, 4)] = 1
        v[idx(4, 0)] = 1
        w[idx(0, 4)] = 1
        add_term(u, v, w)
        
        # i=1, j=4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(1, 4)] = 1
        v[idx(4, 1)] = 1
        w[idx(1, 4)] = 1
        add_term(u, v, w)
        
        # i=2, j=4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(2, 4)] = 1
        v[idx(4, 2)] = 1
        w[idx(2, 4)] = 1
        add_term(u, v, w)
        
        # i=3, j=4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[idx(3, 4)] = 1
        v[idx(4, 3)] = 1
        w[idx(3, 4)] = 1
        add_term(u, v, w)
    
    # Total rank = 19 + 9 = 28
    
    rank = len(U_rows)
    
    U = []
    V = []
    W = []
    for i in range(rank):
        U.append(U_rows[i])
        V.append(V_rows[i])
        W.append(W_rows[i])
    
    return {"U": U, "V": V, "W": W}
