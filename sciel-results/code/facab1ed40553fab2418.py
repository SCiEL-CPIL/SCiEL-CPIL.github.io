import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float)  # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=float)  # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=float)  # 16 x 56
    
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    a1_map = [0, 1, 4, 5]
    a2_map = [2, 3, 6, 7]
    
    def get_b1_map(cp):
        return [0*8+2*cp, 0*8+2*cp+1, 1*8+2*cp, 1*8+2*cp+1]
    
    def get_b2_map(cp):
        return [2*8+2*cp, 2*8+2*cp+1, 3*8+2*cp, 3*8+2*cp+1]
    
    def get_w_map(cp):
        return [2*cp*2+0, (2*cp+1)*2+0, 2*cp*2+1, (2*cp+1)*2+1]
    
    for cp in range(4):
        base_r = cp * 7
        b1_map = get_b1_map(cp)
        w_map = get_w_map(cp)
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a1_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b1_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[w_map[lc], r] += val
    
    for cp in range(4):
        base_r = 28 + cp * 7
        b2_map = get_b2_map(cp)
        w_map = get_w_map(cp)
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la, l]
                if val != 0:
                    U[a2_map[la], r] += val
            for lb in range(4):
                val = V_local[lb, l]
                if val != 0:
                    V[b2_map[lb], r] += val
            for lc in range(4):
                val = W_local[lc, l]
                if val != 0:
                    W[w_map[lc], r] += val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
