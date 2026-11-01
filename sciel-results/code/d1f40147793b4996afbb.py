import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen rank 7 decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: 
    # A: [[a00, a01], [a10, a11]] -> indices 0,1,2,3
    # B: [[b00, b01], [b10, b11]] -> indices 0,1,2,3
    # C: [[c00, c01], [c10, c11]] -> indices 0,1,2,3
    
    # Strassen formulas:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    U_222 = np.array([
        [1, 0, 1, 0, 1, 1, 0], # a00
        [0, 0, 0, 0, 1, 0, 1], # a01
        [0, 1, 0, 0, 0, -1, 0],# a10
        [1, 1, 0, 1, 0, 0, -1] # a11
    ], dtype=float)
    
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 0, 1, 0],  # b01
        [0, 0, -1, 1, 0, 0, 1], # b10
        [1, 0, 0, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, -1], # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Verify 222 decomposition
    # C = sum U[:,r] V[:,r] W[:,r]
    # We construct the full 2x2x6 decomposition by tiling the 222 decomposition.
    # B is 2x6, so it has 3 blocks of 2x2.
    # C is 2x6, so it has 3 blocks of 2x2.
    
    rank = 7 * 3
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for block in range(3):
        offset = block * 7
        
        # U is the same for all blocks (depends only on A)
        U[:, offset:offset+7] = U_222
        
        # V depends on B block.
        # B indices: row j (0,1), col k (0..5).
        # Block 0: k in {0,1}. Local k' in {0,1}.
        # V index in 222 is j*2 + k'.
        # V index in 22x6 is j*6 + k.
        
        # Block 1: k in {2,3}.
        # Block 2: k in {4,5}.
        
        for j in range(2):
            for k_local in range(2):
                v_idx_222 = j * 2 + k_local
                v_idx_full = j * p + (block * 2 + k_local)
                V[v_idx_full, offset:offset+7] = V_222[v_idx_222]
                
        # W depends on C block.
        # C indices: row k (0..5), col i (0,1).
        # Wait, W shape is (p*n, rank).
        # Index is k*n + i.
        # k is the column index of B, which is the row index of C.
        # So for block 'block', k is in {2*block, 2*block+1}.
        
        for k_local in range(2):
            for i in range(2):
                w_idx_222 = k_local * 2 + i
                k_global = block * 2 + k_local
                w_idx_full = k_global * n + i
                W[w_idx_full, offset:offset+7] = W_222[w_idx_222]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
