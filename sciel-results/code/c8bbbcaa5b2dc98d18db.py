"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for evaluator
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor for 2x3x6
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    # Fallback: Schoolbook rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
