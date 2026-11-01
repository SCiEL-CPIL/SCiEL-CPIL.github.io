import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen parameters for 2x2x2
    # U, V, W vectors for the 7 terms
    # Indices for A: 0:00, 1:01, 2:10, 3:11
    # Indices for B: 0:00, 1:01, 2:10, 3:11
    # Indices for C: 0:00, 1:01, 2:10, 3:11
    
    U_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            # Fill U
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            # Fill V
            # Local V indices 0,1,2,3 map to global B indices
            # B is m x p (2 x 8)
            # Global index = row * p + col_idx
            # Local 0: row 0, col 2t -> 0*8 + 2t = 2t
            # Local 1: row 0, col 2t+1 -> 0*8 + 2t+1 = 2t+1
            # Local 2: row 1, col 2t -> 1*8 + 2t = 8+2t
            # Local 3: row 1, col 2t+1 -> 1*8 + 2t+1 = 8+2t+1
            
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            # Fill W
            # Local W indices 0,1,2,3 map to global C indices
            # C is n x p (2 x 8)
            # Convention W[k*n + i, r] where k is col, i is row
            # Local 0: c[0][2t] -> i=0, k=2t -> 2t*2 + 0 = 4t
            # Local 1: c[0][2t+1] -> i=0, k=2t+1 -> (2t+1)*2 + 0 = 4t+2
            # Local 2: c[1][2t] -> i=1, k=2t -> 2t*2 + 1 = 4t+1
            # Local 3: c[1][2t+1] -> i=1, k=2t+1 -> (2t+1)*2 + 1 = 4t+3
            
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
