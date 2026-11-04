import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation
    # 2 blocks of (2x2x2 + 2x2x2 + 2x2x3)
    # 2x2x2 rank 7
    # 2x2x3 rank 11 (7 + 4)
    # Total per block: 7 + 7 + 11 = 25
    # Total rank: 25 * 2 = 50
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add 2x2x2 Strassen
    def add_strassen_222(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # m1 = (a+d)(e+h)
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1 # w00 += m1
        W[w_idx[3]][col] += 1 # w11 += m1
        col += 1
        
        # m2 = (c+d)e
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1 # w10 += m2
        W[w_idx[3]][col] -= 1 # w11 -= m2
        col += 1
        
        # m3 = a(f-h)
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1 # w01 += m3
        W[w_idx[3]][col] += 1 # w11 += m3
        col += 1
        
        # m4 = d(g-e)
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        V[v_idx[0]][col] -= 1
        W[w_idx[0]][col] += 1 # w00 += m4
        W[w_idx[2]][col] += 1 # w10 += m4
        col += 1
        
        # m5 = (a+b)h
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1 # w00 -= m5
        W[w_idx[1]][col] += 1 # w01 += m5
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] -= 1
        V[v_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1 # w11 += m6
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] -= 1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1 # w00 += m7
        col += 1

    # Helper to add 2x2x3 (split into 2x2x2 + 2x2x1)
    def add_223(u_idx, v_idx_2, v_idx_1, w_idx_2, w_idx_1):
        nonlocal col
        # v_idx_2: [v00, v01, v10, v11] for first 2 cols
        # v_idx_1: [v00, v10] for last 1 col (actually v02, v12)
        # w_idx_2: [w00, w01, w10, w11] for first 2 cols
        # w_idx_1: [w00, w10] for last 1 col (w02, w12)
        
        add_strassen_222(u_idx, v_idx_2, w_idx_2)
        
        # 2x2x1 part
        # z0 = a*e + b*f
        # z1 = c*e + d*f
        # U: a, b, c, d
        # V: e, f
        
        # ae
        U[u_idx[0]][col] += 1
        V[v_idx_1[0]][col] += 1
        W[w_idx_1[0]][col] += 1
        col += 1
        
        # bf
        U[u_idx[1]][col] += 1
        V[v_idx_1[1]][col] += 1
        W[w_idx_1[0]][col] += 1
        col += 1
        
        # ce
        U[u_idx[2]][col] += 1
        V[v_idx_1[0]][col] += 1
        W[w_idx_1[1]][col] += 1
        col += 1
        
        # df
        U[u_idx[3]][col] += 1
        V[v_idx_1[1]][col] += 1
        W[w_idx_1[1]][col] += 1
        col += 1

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # A1 indices in U: 0,1,4,5
    u1 = [0, 1, 4, 5]
    
    # B1,1 (cols 0,1) indices in V: 0,1,7,8
    v11 = [0, 1, 7, 8]
    # W indices for C cols 0,1: 0,1,2,3
    w11 = [0, 1, 2, 3]
    add_strassen_222(u1, v11, w11)
    
    # B1,2 (cols 2,3) indices in V: 2,3,9,10
    v12 = [2, 3, 9, 10]
    # W indices for C cols 2,3: 4,5,6,7
    w12 = [4, 5, 6, 7]
    add_strassen_222(u1, v12, w12)
    
    # B1,3 (cols 4,5,6) indices in V: 4,5,6,11,12,13
    # Split into 2x2 (cols 4,5) and 2x1 (col 6)
    v13_2 = [4, 5, 11, 12]
    v13_1 = [6, 13]
    # W indices for C cols 4,5: 8,9,10,11
    w13_2 = [8, 9, 10, 11]
    # W indices for C col 6: 12, 13
    w13_1 = [12, 13]
    add_223(u1, v13_2, v13_1, w13_2, w13_1)

    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # A2 indices in U: 2,3,6,7
    u2 = [2, 3, 6, 7]
    
    # B2,1 (rows 2,3, cols 0,1) indices in V: 14,15,21,22
    v21 = [14, 15, 21, 22]
    # W indices for C cols 0,1: 0,1,2,3
    w21 = [0, 1, 2, 3]
    add_strassen_222(u2, v21, w21)
    
    # B2,2 (rows 2,3, cols 2,3) indices in V: 16,17,23,24
    v22 = [16, 17, 23, 24]
    # W indices for C cols 2,3: 4,5,6,7
    w22 = [4, 5, 6, 7]
    add_strassen_222(u2, v22, w22)
    
    # B2,3 (rows 2,3, cols 4,5,6) indices in V: 18,19,20,25,26,27
    # Split into 2x2 (cols 4,5) and 2x1 (col 6)
    v23_2 = [18, 19, 25, 26]
    v23_1 = [20, 27]
    # W indices for C cols 4,5: 8,9,10,11
    w23_2 = [8, 9, 10, 11]
    # W indices for C col 6: 12, 13
    w23_1 = [12, 13]
    add_223(u2, v23_2, v23_1, w23_2, w23_1)

    return {"U": U, "V": V, "W": W}
