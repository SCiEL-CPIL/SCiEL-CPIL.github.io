import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a column
    def add_column(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    # Helper to add Strassen block
    # i_indices: [i1, i2]
    # j_indices: [j1, j2]
    # k_indices: [k1, k2]
    def add_strassen(i1, i2, j1, j2, k1, k2):
        # M1
        u = {i1*m + j1: 1, i2*m + j2: 1}
        v = {j1*p + k1: 1, j2*p + k2: 1}
        w = {k1*n + i1: 1, k2*n + i2: 1}
        add_column(u, v, w)
        
        # M2
        u = {i2*m + j1: 1, i2*m + j2: 1}
        v = {j1*p + k1: 1}
        w = {k1*n + i2: 1, k2*n + i2: -1}
        add_column(u, v, w)
        
        # M3
        u = {i1*m + j1: 1}
        v = {j1*p + k2: 1, j2*p + k2: -1}
        w = {k2*n + i1: 1, k2*n + i2: 1}
        add_column(u, v, w)
        
        # M4
        u = {i2*m + j2: 1}
        v = {j2*p + k1: 1, j1*p + k1: -1}
        w = {k1*n + i1: 1, k1*n + i2: 1}
        add_column(u, v, w)
        
        # M5
        u = {i1*m + j1: 1, i1*m + j2: 1}
        v = {j2*p + k2: 1}
        w = {k1*n + i1: -1, k2*n + i1: 1}
        add_column(u, v, w)
        
        # M6
        u = {i2*m + j1: 1, i1*m + j1: -1}
        v = {j1*p + k1: 1, j1*p + k2: 1}
        w = {k2*n + i2: 1}
        add_column(u, v, w)
        
        # M7
        u = {i1*m + j2: 1, i2*m + j2: -1}
        v = {j2*p + k1: 1, j2*p + k2: 1}
        w = {k1*n + i1: 1}
        add_column(u, v, w)

    # Part 1: k in {0, 1, 2}
    # 1a. Strassen for j in {0, 1}, k in {0, 1}
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # 1b. Schoolbook for remaining j, k in {0, 1, 2}
    # Pairs: (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)
    for j in [0, 1, 2]:
        for k in [0, 1, 2]:
            if j < 2 and k < 2:
                continue
            for i in range(n):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                u = {u_idx: 1}
                v = {v_idx: 1}
                w = {w_idx: 1}
                add_column(u, v, w)

    # Part 2: k in {3, 4}
    # 2a. Strassen for j in {0, 1}, k in {3, 4}
    add_strassen(0, 1, 0, 1, 3, 4)
    
    # 2b. Schoolbook for remaining j, k in {2} x {3, 4}
    for j in [2]:
        for k in [3, 4]:
            for i in range(n):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                u = {u_idx: 1}
                v = {v_idx: 1}
                w = {w_idx: 1}
                add_column(u, v, w)

    return {"U": U, "V": V, "W": W}
