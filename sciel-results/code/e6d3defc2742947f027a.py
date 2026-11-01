import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 16
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    # Block 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in 0..1, k in 0..1 (cols 0,1)
    # B indices: k in 0..1 (rows 0,1), l in 0..3
    for i in range(2):
        for k in range(2):
            for l in range(4):
                u_idx = i * m + k
                v_idx = k * p + l
                w_idx = l * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    # Block 2: A[:, 2:4] * B[2:4, :]
    # A indices: i in 0..1, k' in 0..1 (cols 2,3)
    # B indices: k' in 0..1 (rows 2,3), l in 0..3
    for i in range(2):
        for k_prime in range(2):
            for l in range(4):
                u_idx = i * m + (2 + k_prime)
                v_idx = (2 + k_prime) * p + l
                w_idx = l * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1

    return {"U": U, "V": V, "W": W}
