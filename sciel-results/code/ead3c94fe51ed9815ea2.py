import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Strassen coefficients for 2x2x2
    U_strassen = [
        np.array([(0, 1), (3, 1)]),
        np.array([(2, 1), (3, 1)]),
        np.array([(0, 1)]),
        np.array([(3, 1)]),
        np.array([(0, 1), (1, 1)]),
        np.array([(2, 1), (0, -1)]),
        np.array([(1, 1), (3, -1)])
    ]
    
    V_strassen = [
        np.array([(0, 1), (3, 1)]),
        np.array([(0, 1)]),
        np.array([(1, 1), (3, -1)]),
        np.array([(2, 1), (0, -1)]),
        np.array([(3, 1)]),
        np.array([(0, 1), (1, 1)]),
        np.array([(2, 1), (3, 1)])
    ]
    
    W_strassen = [
        np.array([(0, 1), (3, 1)]),
        np.array([(2, 1), (3, -1)]),
        np.array([(1, 1), (3, 1)]),
        np.array([(0, 1), (2, 1)]),
        np.array([(0, -1), (1, 1)]),
        np.array([(3, 1)]),
        np.array([(0, 1)])
    ]

    col = 0
    for b in range(3):
        for a in range(2):
            for s in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                
                for idx, coeff in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    u_idx = i_loc * m + (a * 2 + j_loc)
                    U[u_idx, col] += coeff
                
                for idx, coeff in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    v_idx = (a * 2 + j_loc) * p + (b * 2 + k_loc)
                    V[v_idx, col] += coeff
                
                for idx, coeff in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    w_idx = (b * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
