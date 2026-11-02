import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2x2
    # Each term is (u_indices, v_indices, w_indices)
    # u_indices: list of (i, k, coeff)
    # v_indices: list of (k, l, coeff) where l is 0 or 1 (local)
    # w_indices: list of (l, i, coeff) where l is 0 or 1 (local)
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    t_idx = 0
    for q in range(4):
        for term in terms:
            u_list, v_list, w_list = term
            for i, k, c in u_list:
                U[i * m + k, t_idx] = c
            for k, l, c in v_list:
                V[k * p + (2 * q + l), t_idx] = c
            for l, i, c in w_list:
                W[(2 * q + l) * n + i, t_idx] = c
            t_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
