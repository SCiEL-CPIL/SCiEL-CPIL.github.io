"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # We decompose p=15 into 7 blocks of 2 and 1 block of 1.
    # Rank for 2x2x2 is 7 (Strassen).
    # Rank for 2x2x1 is 4 (Schoolbook).
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    # Local factors for M_{2,2,2} (Strassen)
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # Indices: U: 0:a, 1:b, 2:c, 3:d
    # V: 0:x, 1:y, 2:z, 3:w
    # W: 0:C00, 1:C01, 2:C10, 3:C11
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # x
        [0, 0, 1, 0, 0, 0, 1],  # y
        [0, 0, 0, 1, 0, 1, 0],  # z
        [1, 0, -1, 0, 1, 0, 1]  # w
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Local factors for M_{2,2,1} (Schoolbook)
    # U_local: 4x4, V_local: 2x4, W_local: 2x4
    # V indices: 0:x, 1:y
    # W indices: 0:C00, 1:C10
    
    U_school = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1]  # d
    ]
    
    V_school = [
        [1, 0, 1, 0], # x
        [0, 1, 0, 1]  # y
    ]
    
    W_school = [
        [1, 1, 0, 0], # C00
        [0, 0, 1, 1]  # C10
    ]
    
    col = 0
    
    # Add 7 blocks of M_{2,2,2}
    for block in range(7):
        # V global indices for this block
        # x: 0*15 + 2*block
        # y: 0*15 + 2*block + 1
        # z: 1*15 + 2*block
        # w: 1*15 + 2*block + 1
        v_idx = [
            2 * block,
            2 * block + 1,
            15 + 2 * block,
            15 + 2 * block + 1
        ]
        
        # W global indices for this block
        # C00: (2*block)*2 + 0
        # C01: (2*block+1)*2 + 0
        # C10: (2*block)*2 + 1
        # C11: (2*block+1)*2 + 1
        w_idx = [
            (2 * block) * 2 + 0,
            (2 * block + 1) * 2 + 0,
            (2 * block) * 2 + 1,
            (2 * block + 1) * 2 + 1
        ]
        
        # Add factors
        for r in range(7):
            # U
            for i in range(4):
                U[i][col + r] += U_strassen[i][r]
            
            # V
            for j in range(4):
                V[v_idx[j]][col + r] += V_strassen[j][r]
            
            # W
            for k in range(4):
                W[w_idx[k]][col + r] += W_strassen[k][r]
        
        col += 7
    
    # Add 1 block of M_{2,2,1} at column 14
    block = 7 # Not used for index, just logic
    # V global indices
    # x: 0*15 + 14
    # y: 1*15 + 14
    v_idx = [
        14,
        29
    ]
    
    # W global indices
    # C00: 14*2 + 0
    # C10: 14*2 + 1
    w_idx = [
        28,
        29
    ]
    
    for r in range(4):
        # U
        for i in range(4):
            U[i][col + r] += U_school[i][r]
        
        # V
        for j in range(2):
            V[v_idx[j]][col + r] += V_school[j][r]
        
        # W
        for k in range(2):
            W[w_idx[k]][col + r] += W_school[k][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
