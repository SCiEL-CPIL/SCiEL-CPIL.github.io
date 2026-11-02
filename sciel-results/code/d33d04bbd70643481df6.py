"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 35 (7 Strassen terms * 5 vector components)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # u[s][i][j]
    # v[s][j][q]
    # w[s][i][q]
    # Indices 0-based: i,j in {0,1}, q in {0,1}
    
    # M1 = (A11 + A22)(B11 + B22)
    u1 = [[1, 0], [0, 1]]
    v1 = [[1, 0], [0, 1]]
    w1 = [[1, 0], [0, 1]]
    
    # M2 = (A21 + A22)B11
    u2 = [[0, 0], [1, 1]]
    v2 = [[1, 0], [0, 0]]
    w2 = [[0, 0], [1, -1]]
    
    # M3 = A11(B12 - B22)
    u3 = [[1, 0], [0, 0]]
    v3 = [[0, 1], [0, -1]]
    w3 = [[0, 1], [0, 1]]
    
    # M4 = A22(B21 - B11)
    u4 = [[0, 0], [0, 1]]
    v4 = [[-1, 0], [1, 0]]
    w4 = [[1, 0], [1, 0]]
    
    # M5 = (A11 + A12)B22
    u5 = [[1, 1], [0, 0]]
    v5 = [[0, 0], [0, 1]]
    w5 = [[-1, 1], [0, 0]]
    
    # M6 = (A21 - A11)(B11 + B12)
    u6 = [[-1, 0], [1, 0]]
    v6 = [[1, 1], [0, 0]]
    w6 = [[0, 0], [0, 1]]
    
    # M7 = (A12 - A22)(B21 + B22)
    u7 = [[0, 1], [-1, 0]]
    v7 = [[0, 0], [1, 1]]
    w7 = [[1, 0], [0, 0]]
    
    strassen_u = [u1, u2, u3, u4, u5, u6, u7]
    strassen_v = [v1, v2, v3, v4, v5, v6, v7]
    strassen_w = [w1, w2, w3, w4, w5, w6, w7]
    
    # Fill U, V, W
    # c index: 5 * (s-1) + r
    for s in range(7):
        for r in range(5):
            c = 5 * s + r
            u_s = strassen_u[s]
            v_s = strassen_v[s]
            w_s = strassen_w[s]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = u_s[i][j]
                    
            # Fill V
            # V[j * p + k, c]
            # k = 5 * q' + r
            for j in range(2):
                for qp in range(2):
                    k = 5 * qp + r
                    V[j * p + k][c] = v_s[j][qp]
                    
            # Fill W
            # W[k * n + i, c]
            # k = 5 * q + r
            for i in range(2):
                for q in range(2):
                    k = 5 * q + r
                    W[k * n + i][c] = w_s[i][q]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
