"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W rows: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)  (Index k*n+i)
    
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # M1, M2, M3, M4, M5, M6, M7
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V_S = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_S = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ]
    
    # Map to 3 blocks
    for block in range(3):
        offset = block * 7
        
        # U is same for all blocks (shares A)
        for i in range(4):
            for r in range(7):
                U[i][offset + r] = U_S[i][r]
        
        # V mapping
        # Block 0: k=0,1. V rows: 0,1,6,7
        # Block 1: k=2,3. V rows: 2,3,8,9
        # Block 2: k=4,5. V rows: 4,5,10,11
        v_indices = [
            [0, 1, 6, 7],
            [2, 3, 8, 9],
            [4, 5, 10, 11]
        ]
        
        # W mapping
        # Block 0: k=0,1. W rows: 0,1,2,3
        # Block 1: k=2,3. W rows: 4,5,6,7
        # Block 2: k=4,5. W rows: 8,9,10,11
        w_indices = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
        ]
        
        for i in range(4):
            for r in range(7):
                V[v_indices[block][i]][offset + r] = V_S[i][r]
                W[w_indices[block][i]][offset + r] = W_S[i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
