"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145  # Safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Random test matrices
    for trial in range(50):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        C_actual = np.zeros((n, p), dtype=int)
        for r in range(rank):
            Ur = np.array([U[i * m + j][r] for i in range(n) for j in range(m)]).reshape(n, m)
            Vr = np.array([V[j * p + k][r] for j in range(m) for k in range(p)]).reshape(m, p)
            Wr = np.array([W[k * n + i][r] for k in range(p) for i in range(n)]).reshape(n, p)
            C_actual += Ur @ Vr * Wr
        if not np.array_equal(C_expected, C_actual):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 4*4*7 = 112
    # This is guaranteed correct and fits well within the time limit.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
