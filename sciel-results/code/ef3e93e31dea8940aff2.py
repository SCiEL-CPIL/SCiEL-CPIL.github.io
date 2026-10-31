import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients
    # alpha[r][u][v]
    alpha = [
        [[1, 0], [0, 1]],  # r=0: A00 + A11
        [[0, 0], [1, 1]],  # r=1: A10 + A11
        [[1, 0], [0, 0]],  # r=2: A00
        [[0, 0], [0, 1]],  # r=3: A11
        [[1, 1], [0, 0]],  # r=4: A00 + A01
        [[-1, 0], [1, 0]], # r=5: A10 - A00
        [[0, 1], [0, -1]], # r=6: A01 - A11
    ]
    
    beta = [
        [[1, 0], [0, 1]],  # r=0: B00 + B11
        [[1, 0], [0, 0]],  # r=1: B00
        [[0, 1], [0, -1]], # r=2: B01 - B11
        [[-1, 0], [1, 0]], # r=3: B10 - B00
        [[0, 0], [0, 1]],  # r=4: B11
        [[1, 1], [0, 0]],  # r=5: B00 + B01
        [[0, 1], [0, 1]],  # r=6: B10 + B11
    ]
    
    # Inner Strassen coefficients for X (A side)
    # lambda[s][p][q]
    lam = [
        [[1, 0], [0, 1]],  # s=0: x00 + x11
        [[0, 0], [1, 1]],  # s=1: x10 + x11
        [[1, 0], [0, 0]],  # s=2: x00
        [[0, 0], [0, 1]],  # s=3: x11
        [[1, 1], [0, 0]],  # s=4: x00 + x01
        [[-1, 0], [1, 0]], # s=5: x10 - x00
        [[0, 1], [0, -1]], # s=6: x01 - x11
    ]
    
    # Inner Strassen coefficients for Y (B side)
    # mu[s][p][q]
    mu = [
        [[1, 0], [0, 1]],  # s=0: y00 + y11
        [[1, 0], [0, 0]],  # s=1: y00
        [[0, 1], [0, -1]], # s=2: y01 - y11
        [[-1, 0], [1, 0]], # s=3: y10 - y00
        [[0, 0], [0, 1]],  # s=4: y11
        [[1, 1], [0, 0]],  # s=5: y00 + y01
        [[0, 1], [0, 1]],  # s=6: y10 + y11
    ]
    
    # Inner Strassen output weights w[p][q][s]
    # Z_pq = sum_s w[p][q][s] S_s
    w = [
        [ # p=0
            [1, 0, 0, 0, 0, 0, 0], # q=0: S0
            [0, 0, 0, 0, 0, 0, 0], # q=1
        ],
        [ # p=1
            [0, 0, 0, 0, 0, 0, 0], # q=0
            [1, 0, 0, 0, 0, 0, 0], # q=1: S0
        ]
    ]
    # Correcting w based on formulas:
    # Z00 = S0 + S3 - S4 + S6
    # Z01 = S2 + S4
    # Z10 = S1 + S3
    # Z11 = S0 - S1 + S2 + S5
    
    w[0][0] = [1, 0, 0, 1, -1, 0, 1]
    w[0][1] = [0, 0, 1, 0, 1, 0, 0]
    w[1][0] = [0, 1, 0, 1, 0, 0, 0]
    w[1][1] = [1, -1, 1, 0, 0, 1, 0]
    
    # Outer Strassen output weights gamma[u][v][r]
    # C_uv = sum_r gamma[u][v][r] M_r
    gamma = [
        [ # u=0
            [1, 0, 0, 0, 0, 0, 0], # v=0: M0
            [0, 0, 0, 0, 0, 0, 0], # v=1
        ],
        [ # u=1
            [0, 0, 0, 0, 0, 0, 0], # v=0
            [1, 0, 0, 0, 0, 0, 0], # v=1: M0
        ]
    ]
    # Correcting gamma:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    gamma[0][0] = [1, 0, 0, 1, -1, 0, 1]
    gamma[0][1] = [0, 0, 1, 0, 1, 0, 0]
    gamma[1][0] = [0, 1, 0, 1, 0, 0, 0]
    gamma[1][1] = [1, -1, 1, 0, 0, 1, 0]
    
    for r in range(7):
        for s in range(7):
            c = r * 7 + s
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    u_block = i // 2
                    v_block = j // 2
                    p_idx = i % 2
                    q_idx = j % 2
                    val = lam[s][p_idx][q_idx] * alpha[r][u_block][v_block]
                    U[i * 4 + j][c] = val
            
            # Fill V
            for j in range(4):
                for k in range(4):
                    u_block = j // 2
                    v_block = k // 2
                    p_idx = j % 2
                    q_idx = k % 2
                    val = mu[s][p_idx][q_idx] * beta[r][u_block][v_block]
                    V[j * 4 + k][c] = val
            
            # Fill W
            for u in range(2):
                for v in range(2):
                    for p in range(2):
                        for q in range(2):
                            i = u * 2 + p
                            k = v * 2 + q
                            val = gamma[u][v][r] * w[p][q][s]
                            if val != 0:
                                W[k * 4 + i][c] += val
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
