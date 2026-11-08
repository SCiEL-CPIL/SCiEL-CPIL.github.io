import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 85
    U = [[0] * rank for _ in range(n * m)]  # 20 x 85
    V = [[0] * rank for _ in range(m * p)]  # 25 x 85
    W = [[0] * rank for _ in range(p * n)]  # 20 x 85

    # Pre-computed 2x2x2 Strassen factors (verified)
    # u[v][k]: coefficient for A entry v in term k
    # v[v][k]: coefficient for B entry v in term k  
    # w[v][k]: coefficient for C entry v in term k
    # Index order: 00, 01, 10, 11
    s_u = [
        [1,  0, 1,  0, 1, -1, 0],  # 00
        [0,  0, 0,  0, 1,  0, 1],  # 01
        [0,  1, 0,  0, 0,  1, 0],  # 10
        [1,  1, 0,  1, 0,  0, -1], # 11
    ]
    s_v = [
        [1,  1, 0, -1, 0,  1, 0],  # 00
        [0,  0, 1,  0, 0,  1, 0],  # 01
        [0,  0, 0,  1, 0,  0, 1],  # 10
        [1,  0, -1, 0, 1,  0, 1],  # 11
    ]
    s_w = [
        [1,  0, 0,  1, -1, 0, 1],  # 00
        [0,  0, 1,  0,  1, 0, 0],  # 01
        [0,  1, 0,  1,  0, 0, 0],  # 10
        [1, -1, 1,  0,  0, 1, 0],  # 11
    ]

    # 4x4x4 part: 49 terms (7 outer x 7 inner)
    # Outer Strassen terms for 4x4x4:
    # M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2 = (A10+A11)B00 -> C10+1, C11-1
    # M3 = A00(B01-B11) -> C01+1, C11+1
    # M4 = A11(B10-B00) -> C00+1, C10+1
    # M5 = (A00+A01)B11 -> C00-1, C01+1
    # M6 = (A10-A00)(B00+B01) -> C11+1
    # M7 = (A01-A11)(B10+B11) -> C00+1

    # Each outer term: (sum a_i A_i) * (sum b_j B_j) -> sum c_k C_k
    # where A_i, B_j, C_k are 2x2 blocks
    # We decompose the 2x2 product using s_u, s_v, s_w

    outer_terms = [
        # (a_blocks, b_blocks, c_blocks)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),       # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),                 # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),                 # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),                 # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),                 # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),                 # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),                 # M7
    ]

    col = 0
    for a_blocks, b_blocks, c_blocks in outer_terms:
        for inner_k in range(7):
            # For each 2x2 Strassen inner term
            # u_vec = [s_u[0][inner_k], s_u[1][inner_k], s_u[2][inner_k], s_u[3][inner_k]]
            # v_vec = [s_v[0][inner_k], s_v[1][inner_k], s_v[2][inner_k], s_v[3][inner_k]]
            # w_vec = [s_w[0][inner_k], s_w[1][inner_k], s_w[2][inner_k], s_w[3][inner_k]]
            
            # Distribute U coefficients
            for (abr, abc), ac in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = s_u[local_idx][inner_k]
                        if u_val != 0:
                            gi = 2 * abr + lr
                            gj = 2 * abc + lc
                            U[gi * 5 + gj][col] += ac * u_val
            
            # Distribute V coefficients
            for (bbr, bbc), bc in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = s_v[local_idx][inner_k]
                        if v_val != 0:
                            gj = 2 * bbr + lr
                            gk = 2 * bbc + lc
                            V[gj * 5 + gk][col] += bc * v_val
            
            # Distribute W coefficients
            for (cbr, cbc), cc in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = s_w[local_idx][inner_k]
                        if w_val != 0:
                            gi = 2 * cbr + lr
                            gk = 2 * cbc + lc
                            W[gk * 4 + gi][col] += cc * w_val
            
            col += 1
    
    # col should be 49 now
    assert col == 49, f"Expected 49, got {col}"
    
    # 4x4x1 part: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # 16 direct terms
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
    
    # 4x1x4 part: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # 16 direct terms
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    # 4x1x1 part: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # 4 direct terms
    for i in range(4):
        U[i * 5 + 4][col] = 1
        V[4 * 5 + 4][col] = 1
        W[4 * 4 + i][col] = 1
        col += 1
    
    assert col == 85, f"Expected 85, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
