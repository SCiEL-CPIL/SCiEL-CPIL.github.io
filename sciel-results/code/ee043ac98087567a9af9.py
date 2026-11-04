import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for M_2,2,2
    # u, v, w are vectors of length 4
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 1],
        [0, 1, -1, 1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, -1],
        [-1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, -1, 1, 0]
    ]
    
    # M_2,4,6 decomposition
    # A is 2x4 (8 dims). Split into A_L (0-3) and A_R (4-7).
    # B is 4x6 (24 dims). Split into B_L (0-11) and B_R (12-23).
    # C is 2x6 (12 dims).
    
    rank = 42
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    
    # Helper to add a term from M_2,2,6 decomposition
    # a_offset: offset in A (0 for L, 4 for R)
    # b_offset: offset in B (0 for L, 12 for R)
    # c_offset: offset in C (0 for both, as C is shared)
    # block: 0, 1, 2 for the 3 2x2 blocks of the 2x6 matrix
    
    def add_term(a_off, b_off, c_off, block_idx, u, v, w):
        nonlocal col
        # u is length 4, place in U at a_off
        for i in range(4):
            U[a_off + i][col] = u[i]
            
        # v is length 4, place in V at b_off + block_idx * 4
        v_base = b_off + block_idx * 4
        for i in range(4):
            V[v_base + i][col] = v[i]
            
        # w is length 4, place in W at c_off + block_idx * 4
        w_base = c_off + block_idx * 4
        for i in range(4):
            W[w_base + i][col] = w[i]
            
        col += 1

    # Part 1: A_L * B_L
    for block in range(3):
        for r in range(7):
            add_term(0, 0, 0, block, strassen_u[r], strassen_v[r], strassen_w[r])
            
    # Part 2: A_R * B_R
    for block in range(3):
        for r in range(7):
            add_term(4, 12, 0, block, strassen_u[r], strassen_v[r], strassen_w[r])
            
    # Check time
    if time.time() > deadline:
        return None
        
    return {"U": U, "V": V, "W": W}
