import time
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Global dimensions
    # U: (n*m, rank) -> (10, rank)
    # V: (m*p, rank) -> (25, rank)
    # W: (p*n, rank) -> (10, rank)
    
    # We will build lists of rows for U, V, W
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        # u_idx is global index in U (i*m + j)
        # v_idx is global index in V (j*p + k)
        # w_idx is global index in W (k*n + i)
        U_rows.append((u_idx, u_val))
        V_rows.append((v_idx, v_val))
        W_rows.append((w_idx, w_val))
        
    def add_strassen_block(I, J, K):
        # I: list of 2 global i indices
        # J: list of 2 global j indices
        # K: list of 2 global k indices
        # Local indices 0, 1 map to I[0], I[1], etc.
        
        # M1 = (a00 + a11)(b00 + b11)
        # U: (I[0], J[0])=1, (I[1], J[1])=1
        # V: (J[0], K[0])=1, (J[1], K[1])=1
        # W: (K[0], I[0])=1, (K[1], I[1])=1
        u1 = [0]*(n*m)
        v1 = [0]*(m*p)
        w1 = [0]*(p*n)
        u1[I[0]*m + J[0]] = 1
        u1[I[1]*m + J[1]] = 1
        v1[J[0]*p + K[0]] = 1
        v1[J[1]*p + K[1]] = 1
        w1[K[0]*n + I[0]] = 1
        w1[K[1]*n + I[1]] = 1
        U_rows.append(u1); V_rows.append(v1); W_rows.append(w1)
        
        # M2 = (a10 + a11)b00
        u2 = [0]*(n*m)
        v2 = [0]*(m*p)
        w2 = [0]*(p*n)
        u2[I[1]*m + J[0]] = 1
        u2[I[1]*m + J[1]] = 1
        v2[J[0]*p + K[0]] = 1
        w2[K[0]*n + I[1]] = 1
        U_rows.append(u2); V_rows.append(v2); W_rows.append(w2)
        
        # M3 = a00(b01 - b11)
        u3 = [0]*(n*m)
        v3 = [0]*(m*p)
        w3 = [0]*(p*n)
        u3[I[0]*m + J[0]] = 1
        v3[J[0]*p + K[1]] = 1
        v3[J[1]*p + K[1]] = -1
        w3[K[1]*n + I[0]] = 1
        U_rows.append(u3); V_rows.append(v3); W_rows.append(w3)
        
        # M4 = a11(b10 - b00)
        u4 = [0]*(n*m)
        v4 = [0]*(m*p)
        w4 = [0]*(p*n)
        u4[I[1]*m + J[1]] = 1
        v4[J[1]*p + K[0]] = 1
        v4[J[0]*p + K[0]] = -1
        w4[K[0]*n + I[1]] = 1
        U_rows.append(u4); V_rows.append(v4); W_rows.append(w4)
        
        # M5 = (a00 + a01)b11
        u5 = [0]*(n*m)
        v5 = [0]*(m*p)
        w5 = [0]*(p*n)
        u5[I[0]*m + J[0]] = 1
        u5[I[0]*m + J[1]] = 1
        v5[J[1]*p + K[1]] = 1
        w5[K[1]*n + I[0]] = -1
        w5[K[1]*n + I[1]] = 1
        U_rows.append(u5); V_rows.append(v5); W_rows.append(w5)
        
        # M6 = (a10 - a00)(b00 + b01)
        u6 = [0]*(n*m)
        v6 = [0]*(m*p)
        w6 = [0]*(p*n)
        u6[I[1]*m + J[0]] = 1
        u6[I[0]*m + J[0]] = -1
        v6[J[0]*p + K[0]] = 1
        v6[J[0]*p + K[1]] = 1
        w6[K[0]*n + I[1]] = 1
        U_rows.append(u6); V_rows.append(v6); W_rows.append(w6)
        
        # M7 = (a01 - a11)(b10 + b11)
        u7 = [0]*(n*m)
        v7 = [0]*(m*p)
        w7 = [0]*(p*n)
        u7[I[0]*m + J[1]] = 1
        u7[I[1]*m + J[1]] = -1
        v7[J[1]*p + K[0]] = 1
        v7[J[1]*p + K[1]] = 1
        w7[K[1]*n + I[0]] = 1
        U_rows.append(u7); V_rows.append(v7); W_rows.append(w7)

    def add_schoolbook_term(i, j, k):
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[i*m + j] = 1
        v[j*p + k] = 1
        w[k*n + i] = 1
        U_rows.append(u); V_rows.append(v); W_rows.append(w)

    # Define the regions
    # J1 = [0, 1, 2], J2 = [3, 4]
    # K1 = [0, 1, 2], K2 = [3, 4]
    
    # Region 1: J1 x K1 (2x3x3)
    # Sub-block Strassen: J=[0,1], K=[0,1]
    add_strassen_block([0, 1], [0, 1], [0, 1])
    # Remaining in R1:
    # j in [0,1], k=2
    for i in [0, 1]:
        for j in [0, 1]:
            add_schoolbook_term(i, j, 2)
    # j=2, k in [0,1]
    for i in [0, 1]:
        for k in [0, 1]:
            add_schoolbook_term(i, 2, k)
    # j=2, k=2
    for i in [0, 1]:
        add_schoolbook_term(i, 2, 2)
        
    # Region 2: J2 x K2 (2x2x2)
    # Strassen: J=[3,4], K=[3,4]
    add_strassen_block([0, 1], [3, 4], [3, 4])
    
    # Region 3: J1 x K2 (2x3x2)
    # Sub-block Strassen: J=[0,1], K=[3,4]
    add_strassen_block([0, 1], [0, 1], [3, 4])
    # Remaining in R3:
    # j=2, k in [3,4]
    for i in [0, 1]:
        for k in [3, 4]:
            add_schoolbook_term(i, 2, k)
            
    # Region 4: J2 x K1 (2x2x3)
    # Sub-block Strassen: J=[3,4], K=[0,1]
    add_strassen_block([0, 1], [3, 4], [0, 1])
    # Remaining in R4:
    # j in [3,4], k=2
    for i in [0, 1]:
        for j in [3, 4]:
            add_schoolbook_term(i, j, 2)

    # Convert to matrices
    rank = len(U_rows)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        u_row = U_rows[c]
        v_row = V_rows[c]
        w_row = W_rows[c]
        for idx, val in enumerate(u_row):
            if val != 0:
                U[idx][c] = val
        for idx, val in enumerate(v_row):
            if val != 0:
                V[idx][c] = val
        for idx, val in enumerate(w_row):
            if val != 0:
                W[idx][c] = val
                
    return {"U": U, "V": V, "W": W}
