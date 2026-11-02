import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _validate(U, V, W, r):
    """Validate decomposition with random test matrices."""
    n, m, p = DIMENSIONS
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_exp = A @ B
    C_comp = np.zeros((n, p))
    for c in range(r):
        a_c = sum(U[i*m+j][c]*A[i,j] for i in range(n) for j in range(m))
        b_c = sum(V[j*p+k][c]*B[j,k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_comp[i,k] += W[k*n+i][c]*a_c*b_c
    return np.allclose(C_comp, C_exp, atol=1e-9)

def construct():
    """
    Returns a tensor decomposition for 2x5 x 5x5 matrix multiplication.
    Schoolbook rank 50.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135

    rank_sb = n*m*p
    U = [[0]*rank_sb for _ in range(n*m)]
    V = [[0]*rank_sb for _ in range(m*p)]
    W = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    # Verify schoolbook
    if not _validate(U, V, W, rank_sb):
        raise ValueError("Schoolbook validation failed")

    return {"U": U, "V": V, "W": W}
