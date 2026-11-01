"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm on three 2x2 blocks
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A = [[a, b], [c, d]]
    # u1: a
    # u2: c+d
    # u3: a+b
    # u4: d
    # u5: a-d
    # u6: b+c
    # u7: a-c
    u_forms = [
        [1, 0, 0, 0],  # u1
        [0, 0, 1, 1],  # u2
        [1, 1, 0, 0],  # u3
        [0, 0, 0, 1],  # u4
        [1, 0, 0, -1], # u5
        [0, 1, 1, 0],  # u6
        [1, 0, -1, 0]  # u7
    ]
    
    # Strassen linear forms for B (2x2)
    # B = [[e, f], [g, h]]
    # v1: e+h
    # v2: e+f
    # v3: h
    # v4: g-f
    # v5: e+g
    # v6: f+g
    # v7: e-g
    v_forms = [
        [1, 0, 0, 1],  # v1
        [1, 1, 0, 0],  # v2
        [0, 0, 0, 1],  # v3
        [0, 0, 1, -1], # v4
        [1, 0, 1, 0],  # v5
        [0, 1, 1, 0],  # v6
        [1, 0, -1, 0]  # v7
    ]
    
    # Strassen reconstruction for C (2x2)
    # C = [[m00, m01], [m10, m11]]
    # w_k[i] is coefficient of P_k in m_i
    # m00 = P1 + P4 - P5 + P7
    # m01 = P3 + P5
    # m10 = P2 + P4
    # m11 = P1 - P2 + P3 + P6
    # Indices for w_forms: 0:m00, 1:m01, 2:m10, 3:m11
    w_forms = [
        [1, 0, 0, 1],  # P1 coeffs for m00..m11
        [0, 0, 1, -1], # P2 coeffs
        [0, 1, 0, 1],  # P3 coeffs
        [1, 0, 1, 0],  # P4 coeffs
        [-1, 1, 0, 0], # P5 coeffs
        [0, 0, 0, 1],  # P6 coeffs
        [1, 0, 0, 0]   # P7 coeffs
    ]
    
    # Fill U
    for b in range(3):
        for k in range(7):
            col = b * 7 + k
            for i in range(4):
                U[i][col] = u_forms[k][i]
                
    # Fill V
    # B is 2x6. Flattened index j*6 + k.
    # Block b corresponds to columns 2b, 2b+1.
    # e = B[0, 2b] -> idx 2b
    # f = B[0, 2b+1] -> idx 2b+1
    # g = B[1, 2b] -> idx 6+2b
    # h = B[1, 2b+1] -> idx 6+2b+1
    for b in range(3):
        idx_e = 2 * b
        idx_f = 2 * b + 1
        idx_g = 6 + 2 * b
        idx_h = 6 + 2 * b + 1
        for k in range(7):
            col = b * 7 + k
            V[idx_e][col] = v_forms[k][0]
            V[idx_f][col] = v_forms[k][1]
            V[idx_g][col] = v_forms[k][2]
            V[idx_h][col] = v_forms[k][3]
            
    # Fill W
    # C is 2x6. W index is k*n + i = 2*k + i.
    # Block b corresponds to columns 2b, 2b+1.
    # m00 = C[0, 2b] -> idx 2*(2b) + 0 = 4b
    # m01 = C[0, 2b+1] -> idx 2*(2b+1) + 0 = 4b+2
    # m10 = C[1, 2b] -> idx 2*(2b) + 1 = 4b+1
    # m11 = C[1, 2b+1] -> idx 2*(2b+1) + 1 = 4b+3
    for b in range(3):
        idx_m00 = 4 * b
        idx_m01 = 4 * b + 2
        idx_m10 = 4 * b + 1
        idx_m11 = 4 * b + 3
        for k in range(7):
            col = b * 7 + k
            W[idx_m00][col] = w_forms[k][0]
            W[idx_m01][col] = w_forms[k][1]
            W[idx_m10][col] = w_forms[k][2]
            W[idx_m11][col] = w_forms[k][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
