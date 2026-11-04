"""Matrix multiplication 3x3 by 3x7 - rank minimization via structural factorization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    # Build coefficient tensors for each c
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k])
    # Test with several random matrices
    rng = np.random.default_rng(42)
    for trial in range(8):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
            # Linear form of A: for each i, sum_j U[i*m+j, c] * A[i,j]
            # Linear form of B: for each k, sum_j V[j*p+k, c] * B[j,k]
            a_form = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_form[i] += U[i * m + j][c] * A[i, j]
            b_form = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_form[k] += V[j * p + k][c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i][c] * a_form[i] * b_form[k]
        if not np.allclose(C_true, C_comp, atol=tol, rtol=tol):
            return False
    return True


def _schoolbook(n, m, p):
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


def _rank_one(n, m, p):
    """Structural factorization: T[i,j,k] = 1*1*1 factors as rank-1."""
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 120.0  # safety margin below 150s kill

    n, m, p = DIMENSIONS

    # Try rank-1 structural factorization first (fast, O(1) construction)
    candidate = _rank_one(n, m, p)
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate

    # Fallback: schoolbook decomposition (guaranteed valid)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
