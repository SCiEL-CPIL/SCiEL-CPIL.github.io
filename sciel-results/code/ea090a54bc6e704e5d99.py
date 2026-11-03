"""Rewritten program: use a structured blocking strategy with Strassen on 2x2 blocks
and schoolbook on the rest to reduce rank below the schoolbook 75."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block partitioning:
    # A (3x5): 
    #   A11: i=0, j=0,1 (1x2)
    #   A12: i=0, j=2,3,4 (1x3)
    #   A21: i=1,2, j=0,1 (2x2)
    #   A22: i=1,2, j=2,3,4 (2x3)
    
    # B (5x5):
    #   B11: j=0,1, k=0,1 (2x2)
    #   B12: j=0,1, k=2,3,4 (2x3)
    #   B21: j=2,3,4, k=0,1 (3x2)
    #   B22: j=2,3,4, k=2,3,4 (3x3)
    
    # C (3x5):
    #   C11: i=0, k=0,1 (1x2)
    #   C12: i=0, k=2,3,4 (1x3)
    #   C21: i=1,2, k=0,1 (2x2)
    #   C22: i=1,2, k=2,3,4 (2x3)
    
    # Terms:
    # 1. A11 * B11 -> C11 (1x2x2, rank 4)
    # 2. A12 * B21 -> C11 (1x3x2, rank 6)
    # 3. A11 * B12 -> C12 (1x2x3, rank 6)
    # 4. A12 * B22 -> C12 (1x3x3, rank 9)
    # 5. A21 * B11 -> C21 (2x2x2, rank 7 via Strassen)
    # 6. A22 * B21 -> C21 (2x3x2, rank 12)
    # 7. A21 * B12 -> C22 (2x2x3, rank 12)
    # 8. A22 * B22 -> C22 (2x3x3, rank 18)
    
    total_rank = 4 + 6 + 6 + 9 + 7 + 12 + 12 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook_term(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # 1. A11 * B11
    for i in [0]:
        for j in [0, 1]:
            for k in [0, 1]:
                add_schoolbook_term(i, j, k)
                
    # 2. A12 * B21
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                add_schoolbook_term(i, j, k)
                
    # 3. A11 * B12
    for i in [0]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                add_schoolbook_term(i, j, k)
                
    # 4. A12 * B22
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                add_schoolbook_term(i, j, k)
                
    # 5. A21 * B11 (Strassen)
    # Local indices for A21: i_loc in {0,1} (global i in {1,2}), j_loc in {0,1} (global j in {0,1})
    # Local indices for B11: j_loc in {0,1} (global j in {0,1}), k_loc in {0,1} (global k in {0,1})
    # Local indices for C21: i_loc in {0,1} (global i in {1,2}), k_loc in {0,1} (global k in {0,1})
    
    def get_global_U(i_loc, j_loc):
        i = i_loc + 1
        j = j_loc
        return i * m + j
        
    def get_global_V(j_loc, k_loc):
        j = j_loc
        k = k_loc
        return j * p + k
        
    def get_global_W(i_loc, k_loc):
        i = i_loc + 1
        k = k_loc
        return k * n + i

    # Strassen terms definitions:
    # (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts: {local_idx: val}
    
    # m1 = (A00 + A11)(B00 + B11)
    # W: C00 +1, C11 +1
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    
    # m2 = (A10 + A11)B00
    # W: C10 +1, C11 -1
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    
    # m3 = A00(B01 - B11)
    # W: C01 +1, C11 +1
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    
    # m4 = A11(B10 - B00)
    # W: C00 +1, C10 +1
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    
    # m5 = (A00 + A01)B11
    # W: C00 -1, C01 +1
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    
    # m6 = (A10 - A00)(B00 + B01)
    # W: C11 +1
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    
    # m7 = (A01 - A11)(B10 + B11)
    # W: C00 +1
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for u, v, w in strassen_terms:
        for (i_l, j_l), val_u in u.items():
            idx_u = get_global_U(i_l, j_l)
            U[idx_u][col] = val_u
        for (j_l, k_l), val_v in v.items():
            idx_v = get_global_V(j_l, k_l)
            V[idx_v][col] = val_v
        for (i_l, k_l), val_w in w.items():
            idx_w = get_global_W(i_l, k_l)
            W[idx_w][col] = val_w
        col += 1
        
    # 6. A22 * B21
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                add_schoolbook_term(i, j, k)
                
    # 7. A21 * B12
    for i in [1, 2]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                add_schoolbook_term(i, j, k)
                
    # 8. A22 * B22
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                add_schoolbook_term(i, j, k)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
