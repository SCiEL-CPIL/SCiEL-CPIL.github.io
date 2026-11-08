import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank) = (6, 22)
    # V shape (m*p, rank) = (12, 22)
    # W shape (p*n, rank) = (8, 22)
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen terms
    def add_strassen(u_idx, v_idx, w_idx, start_col):
        # u_idx: [A00, A01, A10, A11]
        # v_idx: [B00, B01, B10, B11]
        # w_idx: [C00, C01, C10, C11]
        
        # M1 = (A00 + A11)(B00 + B11)
        col = start_col
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1 # C00
        W[w_idx[3]][col] += 1 # C11
        col += 1
        
        # M2 = (A10 + A11)B00
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1 # C10
        col += 1
        
        # M3 = A00(B01 - B11)
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[3]][col] += -1
        W[w_idx[1]][col] += 1 # C01
        W[w_idx[3]][col] += 1 # C11
        col += 1
        
        # M4 = A11(B10 - B00)
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        V[v_idx[0]][col] += -1
        W[w_idx[0]][col] += 1 # C00
        W[w_idx[2]][col] += 1 # C10
        col += 1
        
        # M5 = (A00 + A01)B11
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[1]][col] += 1 # C01
        W[w_idx[0]][col] += -1 # C00
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] += -1
        V[v_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] += -1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1 # C00
        W[w_idx[1]][col] += -1 # C01
        col += 1
        
        return col

    # Helper to add schoolbook terms for 2x1 * 1x4
    def add_schoolbook_2x1x4(u_idx, v_idx, w_idx, start_col):
        # u_idx: [A0, A1]
        # v_idx: [B0, B1, B2, B3]
        # w_idx: [C00, C01, C02, C03, C10, C11, C12, C13]
        # w_idx order: i=0, k=0..3 then i=1, k=0..3
        
        col = start_col
        for i in range(2):
            for k in range(4):
                U[u_idx[i]][col] += 1
                V[v_idx[k]][col] += 1
                W[w_idx[i*4 + k]][col] += 1
                col += 1
        return col

    # Define indices
    # A: 2x3. i in 0..1, j in 0..2. idx = i*3 + j
    # A1 (2x2): i in 0..1, j in 0..1
    u_a1 = [0, 1, 3, 4] # A00, A01, A10, A11
    # A2 (2x1): i in 0..1, j=2
    u_a2 = [2, 5] # A02, A12
    
    # B: 3x4. j in 0..2, k in 0..3. idx = j*4 + k
    # B1a (2x2): j in 0..1, k in 0..1
    v_b1a = [0, 1, 4, 5] # B00, B01, B10, B11
    # B1b (2x2): j in 0..1, k in 2..3
    v_b1b = [2, 3, 6, 7] # B02, B03, B12, B13
    # B2 (1x4): j=2, k in 0..3
    v_b2 = [8, 9, 10, 11] # B20, B21, B22, B23
    
    # C: 2x4. i in 0..1, k in 0..3. idx = k*2 + i
    # C1a (2x2): i in 0..1, k in 0..1
    # C00 (k=0,i=0) -> 0
    # C01 (k=1,i=0) -> 2
    # C10 (k=0,i=1) -> 1
    # C11 (k=1,i=1) -> 3
    w_c1a = [0, 2, 1, 3]
    
    # C1b (2x2): i in 0..1, k in 2..3
    # C02 (k=2,i=0) -> 4
    # C03 (k=3,i=0) -> 6
    # C12 (k=2,i=1) -> 5
    # C13 (k=3,i=1) -> 7
    w_c1b = [4, 6, 5, 7]
    
    # C2 (2x4): i in 0..1, k in 0..3
    # Order: i=0, k=0..3; i=1, k=0..3
    # 0, 2, 4, 6, 1, 3, 5, 7
    w_c2 = [0, 2, 4, 6, 1, 3, 5, 7]
    
    col = 0
    col = add_strassen(u_a1, v_b1a, w_c1a, col)
    col = add_strassen(u_a1, v_b1b, w_c1b, col)
    col = add_schoolbook_2x1x4(u_a2, v_b2, w_c2, col)
    
    return {"U": U, "V": V, "W": W}

