import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 128
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_idx, w_val):
        nonlocal col
        if col >= rank:
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        W[w_idx][col] += w_val
        col += 1

    # Strassen definitions
    X_defs = [
        [[(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]],
        [[(1,0,1), (1,1,1)], [(1,0,1), (1,1,1)]],
        [[(0,0,1)], [(0,0,1)]],
        [[(1,1,1)], [(1,1,1)]],
        [[(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]],
        [[(0,1,1), (0,0,-1)], [(0,1,1), (0,0,-1)]],
        [[(1,1,1), (1,0,-1)], [(1,1,1), (1,0,-1)]]
    ]
    
    Y_defs = [
        [[(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]],
        [[(0,0,1)], [(0,0,1)]],
        [[(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]],
        [[(1,0,1), (0,0,-1)], [(1,0,1), (0,0,-1)]],
        [[(1,1,1)], [(1,1,1)]],
        [[(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)]],
        [[(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)]]
    ]

    contrib_map = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(1, 0, 1), (1, 1, -1)],
        2: [(1, 1, 1)],
        3: [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        4: [(0, 1, 1)],
        5: [(0, 0, -1)],
        6: [(0, 0, -1), (0, 1, 1)]
    }

    def process_strassen_part(k_offset):
        nonlocal col
        for m in range(7):
            for u in range(2):
                for v in range(2):
                    for l in range(2):
                        u_indices = []
                        u_vals = []
                        for (bI, bJ, sign) in X_defs[m][u][l]:
                            a_idx = (2 * bI + u) * m + (2 * bJ + l)
                            u_indices.append(a_idx)
                            u_vals.append(sign)
                        
                        v_indices = []
                        v_vals = []
                        for (bJ, bK, sign) in Y_defs[m][l][v]:
                            b_idx = (2 * bJ + l) * p + (2 * bK + v + k_offset)
                            v_indices.append(b_idx)
                            v_vals.append(sign)
                        
                        for (I, K, S) in contrib_map[m]:
                            w_idx = (2 * K + v + k_offset) * n + (2 * I + u)
                            w_val = S
                            add_term(u_indices, u_vals, v_indices, v_vals, w_idx, w_val)

    process_strassen_part(0)
    process_strassen_part(4)
    
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            v_idx = j * p + 8
            w_idx = 8 * n + i
            add_term([u_idx], [1.0], [v_idx], [1.0], w_idx, 1.0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
