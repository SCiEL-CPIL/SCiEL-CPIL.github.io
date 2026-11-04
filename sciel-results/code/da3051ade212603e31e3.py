"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Base 2x2 Strassen vectors
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    u_base = [
        [1, 0, 0, 1], # s=1
        [0, 0, 1, 1], # s=2
        [1, 0, 0, 0], # s=3
        [0, 0, 0, 1], # s=4
        [1, 1, 0, 0], # s=5
        [-1, 0, 1, 0],# s=6
        [0, 1, 0, -1] # s=7
    ]
    v_base = [
        [1, 0, 0, 1], # s=1
        [1, 0, 0, 0], # s=2
        [0, 1, 0, -1],# s=3
        [0, 0, 1, -1],# s=4
        [0, 0, 0, 1], # s=5
        [1, 1, 0, 0], # s=6
        [0, 0, 1, 1]  # s=7
    ]
    w_base = [
        [1, 0, 0, 1], # s=1
        [0, 0, 1, 0], # s=2
        [0, 1, 0, 0], # s=3
        [1, 0, 0, 0], # s=4
        [-1, 1, 0, 0],# s=5
        [0, 0, 0, 1], # s=6
        [1, 0, 0, 0]  # s=7
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Construct rank-49 decomposition
    for s1 in range(7):
        for s2 in range(7):
            r = s1 * 7 + s2
            # Loop over inner indices i2, j2, k2 (0,1) and outer i1, j1, k1 (0,1)
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        # U part: depends on i1, j1 (outer) and i2, j2 (inner)
                        # Wait, U depends on i, j.
                        # i = 2*i1 + i2, j = 2*j1 + j2.
                        # Coeff = u_base[s1][i1*2+j1] * u_base[s2][i2*2+j2]
                        for i2 in range(2):
                            for j2 in range(2):
                                i_idx = 2 * i1 + i2
                                j_idx = 2 * j1 + j2
                                u_idx = i_idx * 4 + j_idx
                                coeff_u = u_base[s1][i1 * 2 + j1] * u_base[s2][i2 * 2 + j2]
                                if coeff_u != 0:
                                    U[u_idx][r] += coeff_u
                            
                        # V part: depends on j1, k1 (outer) and j2, k2 (inner)
                        # j = 2*j1 + j2, k = 2*k1 + k2
                        for j2 in range(2):
                            for k2 in range(2):
                                j_idx = 2 * j1 + j2
                                k_idx = 2 * k1 + k2
                                v_idx = j_idx * 4 + k_idx
                                coeff_v = v_base[s1][j1 * 2 + k1] * v_base[s2][j2 * 2 + k2]
                                if coeff_v != 0:
                                    V[v_idx][r] += coeff_v
                                    
                        # W part: depends on k1, i1 (outer) and k2, i2 (inner)
                        # k = 2*k1 + k2, i = 2*i1 + i2
                        for k2 in range(2):
                            for i2 in range(2):
                                k_idx = 2 * k1 + k2
                                i_idx = 2 * i1 + i2
                                w_idx = k_idx * 4 + i_idx
                                coeff_w = w_base[s1][k1 * 2 + i1] * w_base[s2][k2 * 2 + i2]
                                if coeff_w != 0:
                                    W[w_idx][r] += coeff_w

    # Check time
    if time.time() > deadline:
         return {"U": U, "V": V, "W": W}
         
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
