"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen vectors for 2x2 multiplication
    # Index order: (0,0), (0,1), (1,0), (1,1)
    local_U = [
        [1, 0, 0, 1], # L=0
        [0, 0, 1, 1], # L=1
        [1, 0, 0, 0], # L=2
        [0, 0, 0, 1], # L=3
        [1, 1, 0, 0], # L=4
        [-1, 0, 1, 0],# L=5
        [0, 1, 0, -1] # L=6
    ]
    local_V = [
        [1, 0, 0, 1], # L=0
        [1, 0, 0, 0], # L=1
        [0, 1, 0, -1],# L=2
        [0, 0, 1, -1],# L=3
        [0, 0, 0, 1], # L=4
        [1, 1, 0, 0], # L=5
        [0, 0, 1, 1]  # L=6
    ]
    local_W = [
        [1, 0, 0, 1], # L=0
        [0, 0, 1, -1],# L=1
        [0, 1, 0, 1], # L=2
        [1, 0, 1, 0], # L=3
        [-1, 1, 0, 0],# L=4
        [0, 0, 0, 1], # L=5
        [1, 0, 0, -1] # L=6
    ]
    
    # Block level coefficients
    # alpha[K][r][s]
    alpha = [
        [[1, 0], [0, 1]], # K=0
        [[0, 0], [1, 1]], # K=1
        [[1, 0], [0, 0]], # K=2
        [[0, 0], [0, 1]], # K=3
        [[1, 1], [0, 0]], # K=4
        [[-1, 0], [1, 0]],# K=5
        [[0, 1], [0, -1]] # K=6
    ]
    
    # beta[K][u][v]
    beta = [
        [[1, 0], [0, 1]], # K=0
        [[1, 0], [0, 0]], # K=1
        [[0, 1], [0, -1]],# K=2
        [[-1, 0], [1, 0]],# K=3
        [[0, 0], [0, 1]], # K=4
        [[1, 1], [0, 0]], # K=5
        [[0, 0], [1, 1]]  # K=6
    ]
    
    # gamma[K][r][s]
    gamma = [
        [[1, 0], [0, 1]], # K=0
        [[0, 0], [1, -1]],# K=1
        [[0, 1], [0, 1]], # K=2
        [[1, 0], [1, 0]], # K=3
        [[-1, 1], [0, 0]],# K=4
        [[0, 0], [0, 1]], # K=5
        [[1, 0], [0, -1]] # K=6
    ]
    
    for K in range(7):
        for L in range(7):
            col = K * 7 + L
            for i in range(n):
                for j in range(m):
                    r = i // 2
                    s = j // 2
                    ip = i % 2
                    jp = j % 2
                    u_val = alpha[K][r][s] * local_U[L][ip * 2 + jp]
                    if u_val != 0:
                        U[i * m + j][col] += u_val
            
            for j in range(m):
                for k in range(p):
                    u = j // 2
                    v = k // 2
                    jp = j % 2
                    kp = k % 2
                    v_val = beta[K][u][v] * local_V[L][jp * 2 + kp]
                    if v_val != 0:
                        V[j * p + k][col] += v_val
            
            for k in range(p):
                for i in range(n):
                    r = i // 2
                    s = k // 2
                    ip = i % 2
                    kp = k % 2
                    w_val = gamma[K][r][s] * local_W[L][ip * 2 + kp]
                    if w_val != 0:
                        W[k * n + i][col] += w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
