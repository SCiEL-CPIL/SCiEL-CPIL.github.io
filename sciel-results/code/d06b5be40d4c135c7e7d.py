"""Tensor rank minimization for 4x5 by 5x6 matrix multiplication.

Returns an exact decomposition of the matmul tensor into rank-1 terms.
The seed schoolbook has rank 120; we attempt to find a lower-rank exact
decomposition via a bounded search, defaulting to schoolbook.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = None

def _build_target():
    n, m, p = DIMENSIONS
    # T[i, j, k] = 1 for valid triple (i in [0,n), j in [0,m), k in [0,p))
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(n and m):  # placeholder
            pass
    # correct loops
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _matrices_from_factors(U, V, W):
    """Reconstruct the tensor from factor matrices (list-of-lists or arrays)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=float)
    # U has shape (n*m, rank): U[i*m + j, c]
    # V has shape (m*p, rank): V[j*p + k, c]
    # W has shape (p*n, rank): W[k*n + i, c]
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    for c in range(rank):
        T += Uc[:, :, c] * Vc[:, :, c] * Wc[:, :, c].T  # Wc is (p,n); need (i,j,k)
    # Wc[i index is last dim] -> Wc[k, i]; we need (i, j, k):
    # redo properly
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        u = Uc[:, :, c]   # (n, m)
        v = Vc[:, :, c]   # (m, p)
        w = Wc[:, c]      # (p,) -> w[k]
        # term[i,j,k] = u[i,j] * v[j,k] * w[k]
        T += u[:, :, None] * v * w[None, None, :]
    return T

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    T = _matrices_from_factors(U, V, W)
    # target is all-ones on support (which is the full box)
    target = np.ones((n, m, p), dtype=float)
    if T.shape != target.shape:
        return False
    if not np.allclose(T, target, atol=tol, rtol=0):
        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _search_lower_rank(budget):
    """Attempt to find a lower-rank exact decomposition via randomized linear
    construction with exact integer verification. Returns dict or None."""
    n, m, p = DIMENSIONS
    best = None
    # Try a small set of candidate ranks below schoolbook.
    for rank in range(120, 95, -1):
        if time.time() > budget:
            break
        # Randomized construction: pick random sparse factor matrices with
        # small integer entries and check exactness (rare to hit, but bounded).
        # This is a heuristic; we keep attempts bounded.
        attempts = 0
        while attempts < 200 and time.time() < budget:
            attempts += 1
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            T = _matrices_from_factors(U.tolist(), V.tolist(), W.tolist())
            if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
    return best

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 140  # safety margin before 150s kill
    # Baseline: guaranteed-valid schoolbook.
    result = _schoolbook()
    # Attempt bounded search for a lower rank.
    try:
        found = _search_lower_rank(_DEADLINE)
        if found is not None and _verify(found["U"], found["V"], found["W"]):
            result = found
    except Exception:
        pass
    # Final safety: ensure we return a valid decomposition.
    if not _verify(result["U"], result["V"], result["W"]):
        result = _schoolbook()
    return result
# EVOLVE-BLOCK-END
