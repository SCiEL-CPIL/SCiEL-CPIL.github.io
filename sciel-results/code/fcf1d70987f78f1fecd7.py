"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank for 6x6 using Block-Strassen (2x2 blocks)
    # 3x3 block grid, each 2x2 multiplication uses Strassen (rank 7)
    rank = 3 * 3 * 3 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # u_s are coefficients for X (A block), v_s for Y (B block)
    # c_uv_s are coefficients for result Z (C block) entry (u,v) from product P_s
    # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    # P1 = (x00 + x11)(y00 + y11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    c1 = [[1, 0], [0, 1]] # c1[u][v]
    
    # P2 = (x10 + x11)y00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    c2 = [[0, 0], [1, -1]]
    
    # P3 = x00(y01 - y11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    c3 = [[0, 1], [0, 1]]
    
    # P4 = x11(y10 - y00)
    u4 = [0, 0, 0, 1]
    v4 = [-1, 0, 1, 0]
    c4 = [[1, 0], [1, 0]]
    
    # P5 = (x00 + x01)y11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    c5 = [[-1, 1], [0, 0]]
    
    # P6 = (x10 - x00)(y00 + y01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    c6 = [[0, 0], [0, 1]]
    
    # P7 = (x01 - x11)(y10 + y11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    c7 = [[1, 0], [0, 0]]
    
    strassen_terms = [
        (u1, v1, c1),
        (u2, v2, c2),
        (u3, v3, c3),
        (u4, v4, c4),
        (u5, v5, c5),
        (u6, v6, c6),
        (u7, v7, c7)
    ]
    
    col = 0
    for i in range(3): # Block row of C and A
        for j in range(3): # Block col of C and B
            for k in range(3): # Block col of A and Block row of B
                # This combination (i, j, k) contributes to C[i][j] via A[i][k] * B[k][j]
                for s_idx in range(7):
                    u_s, v_s, c_s = strassen_terms[s_idx]
                    
                    # Fill U
                    # A[i][k] is a 2x2 block.
                    # Global row I = 2*i + a, Global col J = 2*k + b
                    for a in range(2):
                        for b in range(2):
                            idx_u = a * 2 + b # 0:00, 1:01, 2:10, 3:11
                            val = u_s[idx_u]
                            if val != 0:
                                global_row = 2 * i + a
                                global_col = 2 * k + b
                                U[global_row * 6 + global_col][col] += val
                                
                    # Fill V
                    # B[k][j] is a 2x2 block.
                    # Global row I = 2*k + c, Global col J = 2*j + d
                    for c in range(2):
                        for d in range(2):
                            idx_v = c * 2 + d
                            val = v_s[idx_v]
                            if val != 0:
                                global_row = 2 * k + c
                                global_col = 2 * j + d
                                V[global_row * 6 + global_col][col] += val
                                
                    # Fill W
                    # C[i][j] is a 2x2 block.
                    # Global row I = 2*i + u, Global col K = 2*j + v
                    # W index is K * 6 + I
                    for u in range(2):
                        for v in range(2):
                            val = c_s[u][v]
                            if val != 0:
                                global_row = 2 * i + u
                                global_col = 2 * j + v
                                W[global_col * 6 + global_row][col] += val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
