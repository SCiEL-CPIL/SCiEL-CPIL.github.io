import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Total rank for schoolbook
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term([i * m + j], [1], 
                         [j * p + k], [1],
                         [k * n + i], [1])

    return {"U": U, "V": V, "W": W}
