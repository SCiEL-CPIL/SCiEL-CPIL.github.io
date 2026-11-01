"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 using Strassen's algorithm for the 2x2 block structure
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[s][i] for A (4 entries)
    # beta[s][j] for B (4 blocks)
    # w[s][k] for C (4 blocks)
    
    alphas = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    betas = [
        [1, 1, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    ws = [
        [1, 0, 0, -1],  # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [1, 0, 0, 0],   # M5
        [0, 0, 0, 1],   # M6
        [-1, 1, 0, -1]  # M7
    ]
    
    for s in range(7):
        for t in range(8):
            r = s * 8 + t
            
            # Fill U
            for i in range(4):
                U[i][r] = alphas[s][i]
                
            # Fill V
            # B blocks: 11 (0,0), 12 (0,8), 21 (1,0), 22 (1,8)
            # V row index: j * 16 + k
            V[t][r] = betas[s][0]       # B11, t
            V[8 + t][r] = betas[s][1]   # B12, t
            V[16 + t][r] = betas[s][2]  # B21, t
            V[24 + t][r] = betas[s][3]  # B22, t
            
            # Fill W
            # C blocks: 11 (0,0), 12 (0,8), 21 (1,0), 22 (1,8)
            # W row index: k * n + i = k * 2 + i
            W[2 * t][r] = ws[s][0]           # C11, t -> k=t, i=0
            W[2 * (8 + t)][r] = ws[s][1]     # C12, t -> k=8+t, i=0
            W[2 * t + 1][r] = ws[s][2]       # C21, t -> k=t, i=1
            W[2 * (8 + t) + 1][r] = ws[s][3] # C22, t -> k=8+t, i=1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
