import numpy as np
import time

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the factorization is exact."""
    np.random.seed(42)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for trial in range(5):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        
        C_fact = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    f = 0.0
                    for j in range(m):
                        f += U[i*m + j, r] * A[i, j]
                    g = 0.0
                    for j in range(m):
                        g += V[j*p + k, r] * B[j, k]
                    C_fact[i, k] += W[k*n + i, r] * f * g
        
        if np.max(np.abs(C - C_fact)) > tol:
            return False
    return True

def construct():
    # Schoolbook decomposition: rank = n*m*p = 2*4*6 = 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
