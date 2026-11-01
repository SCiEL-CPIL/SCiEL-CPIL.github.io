import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 12 decomposition for 2x2x4
    # U is 4x12, V is 8x12, W is 8x12
    # Indices for A: 0:00, 1:01, 2:10, 3:11
    # Indices for B: 0:00, 1:01, 2:10, 3:11, 4:02, 5:03, 6:12, 7:13
    # Indices for C: 0:00, 1:01, 2:10, 3:11, 4:02, 5:03, 6:12, 7:13
    # Note: W index convention is k*n + i.
    # For 2x4, k is 0..3, i is 0..1.
    # W indices:
    # c[0][0] -> 0*2+0=0
    # c[0][1] -> 1*2+0=2
    # c[1][0] -> 0*2+1=1
    # c[1][1] -> 1*2+1=3
    # c[0][2] -> 2*2+0=4
    # c[0][3] -> 3*2+0=6
    # c[1][2] -> 2*2+1=5
    # c[1][3] -> 3*2+1=7
    
    U_12 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1]
    ]
    
    # V and W for 2x2x4
    # This part is risky without verification.
    # I will use the Strassen-based rank 14 for 2x2x4 to be safe?
    # Rank 14 for 2x2x4 -> Rank 28 for 2x2x8. Same as current.
    # I will stick to the current rank 28 solution to ensure correctness.
    
    # Reverting to current logic for safety
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(n * m):
                U[i][col] = U_strassen[r][i]
                
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l]][col] = V_strassen[r][l]
                
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l]][col] = W_strassen[r][l]
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
