import time
import numpy as np
import itertools

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    bs = 3

    # Strassen steps for 2x2 block decomposition
    steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Pre-compute all column mappings
    col_map = {}
    for step_idx, (a_terms, b_terms, c_terms) in enumerate(steps):
        for r in range(bs):
            for s in range(bs):
                for t in range(bs):
                    col = step_idx * 27 + (r * 9 + s * 3 + t)
                    col_map[col] = (step_idx, r, s, t)

    for col, (step_idx, r, s, t) in col_map.items():
        a_terms, b_terms, c_terms = steps[step_idx]

        for I, J, ca in a_terms:
            u_idx = (I * bs + r) * dim + (J * bs + s)
            U[u_idx, col] += ca

        for J, K, cb in b_terms:
            v_idx = (J * bs + s) * dim + (K * bs + t)
            V[v_idx, col] += cb

        for I, K, cc in c_terms:
            w_idx = (K * bs + t) * dim + (I * bs + r)
            W[w_idx, col] += cc

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
