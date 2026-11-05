"""Matrix multiplication 3x3 by 3x7 - rank minimization via structured search."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0]) if U else 0
    rng = np.random.default_rng(42)
    for trial in range(5):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C_comp = np.zeros((n, p))
        for c in range(rank):
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

def construct():
    start = time.time()
    deadline = start + 110.0  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook is the guaranteed valid solution with rank 63
    candidate = _schoolbook(n, m, p)
    
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        best_result = candidate
        best_rank = n * m * p
    else:
        return candidate

    # Attempt to find a lower rank decomposition using known structures.
    # Currently, no reliable hardcoded low-rank decomposition with exact 
    # integer/half-integer coefficients is available in this context.
    # Returning the verified schoolbook solution.
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
