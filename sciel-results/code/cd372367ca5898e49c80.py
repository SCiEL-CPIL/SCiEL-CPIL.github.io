import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Rank 56 decomposition:
    # Split A (2x4) into A1 (2x2), A2 (2x2)
    # Split B (4x8) into B1 (2x8), B2 (2x8)
    # C = A1*B1 + A2*B2
    # Each 2x2 * 2x8 is decomposed into 4 independent 2x2 * 2x2 Strassen blocks.
    # Rank 28 per block, total 56.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=int) # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=int) # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=int) # 16 x 56
    
    # Strassen 2x2 * 2x2 factors
    # U_local[4][7], V_local[4][7], W_local[4][7]
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    # r=0 (M1)
    U_local[0, 0] = 1; U_local[3, 0] = 1 
    V_local[0, 0] = 1; V_local[3, 0] = 1 
    W_local[0, 0] = 1 
    W_local[3, 0] = 1 
    
    # r=1 (M2)
    U_local[2, 1] = 1; U_local[3, 1] = 1 
    V_local[0, 1] = 1 
    W_local[2, 1] = 1 
    W_local[3, 1] = -1 
    
    # r=2 (M3)
    U_local[0, 2] = 1 
    V_local[1, 2] = 1; V_local[3, 2] = -1 
    W_local[1, 2] = 1 
    W_local[3, 2] = 1 
    
    # r=3 (M4)
    U_local[3, 3] = 1 
    V_local[2, 3] = 1; V_local[0, 3] = -1 
    W_local[0, 3] = 1 
    W_local[2, 3] = 1 
    
    # r=4 (M5)
    U_local[0, 4] = 1; U_local[1, 4] = 1 
    V_local[3, 4] = 1 
    W_local[0, 4] = -1 
    W_local[1, 4] = 1 
    
    # r=5 (M6)
    U_local[2, 5] = 1; U_local[0, 5] = -1 
    V_local[0, 5] = 1; V_local[1, 5] = 1 
    W_local[3, 5] = 1 
    
    # r=6 (M7)
    U_local[1, 6] = 1; U_local[3, 6] = -1 
    V_local[2, 6] = 1; V_local[3, 6] = 1 
    W_local[0, 6] = 1 
    
    # Helper to fill global matrices
    def fill_block(U_g, V_g, W_g, U_l, V_l, W_l, a_map, b_map, c_map, base_r):
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_l[la, l]
                if val != 0:
                    U_g[a_map[la], r] += val
            for lb in range(4):
                val = V_l[lb, l]
                if val != 0:
                    V_g[b_map[lb], r] += val
            for lc in range(4):
                val = W_l[lc, l]
                if val != 0:
                    W_g[c_map[lc], r] += val

    # Block 1: A1*B1
    a1_map = [0, 1, 4, 5]
    for c_pair in range(4):
        base_r = c_pair * 7
        b1_map = [
            0*8 + 2*c_pair,
            0*8 + 2*c_pair+1,
            1*8 + 2*c_pair,
            1*8 + 2*c_pair+1
        ]
        c1_map = [
            2*c_pair*2 + 0,
            (2*c_pair+1)*2 + 0,
            2*c_pair*2 + 1,
            (2*c_pair+1)*2 + 1
        ]
        fill_block(U, V, W, U_local, V_local, W_local, a1_map, b1_map, c1_map, base_r)

    # Block 2: A2*B2
    a2_map = [2, 3, 6, 7]
    for c_pair in range(4):
        base_r = 28 + c_pair * 7
        b2_map = [
            2*8 + 2*c_pair,
            2*8 + 2*c_pair+1,
            3*8 + 2*c_pair,
            3*8 + 2*c_pair+1
        ]
        c2_map = [
            2*c_pair*2 + 0,
            (2*c_pair+1)*2 + 0,
            2*c_pair*2 + 1,
            (2*c_pair+1)*2 + 1
        ]
        fill_block(U, V, W, U_local, V_local, W_local, a2_map, b2_map, c2_map, base_r)

    # Check time limit
    if time.time() - start_time > 115:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist}
# EVOLVE-BLOCK-END
