"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication."""
import random
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
_DEADLINE = None
_BEST = None
_BEST_RANK = None


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify(U, V, W, T, rng, trials=8):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # Exact tensor check
    recon = np.einsum("ac,bc,dc->abd", U, V, W)
    if not np.array_equal(recon, T):
        return False
    # Functional check with random integer inputs
    for _ in range(trials):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C = np.zeros((n, p), dtype=np.int64)
        for c in range(R):
            C += np.outer(U[:, c].T @ A @ V[:, c], W[:, c].T @ B)
        if not np.array_equal(C, C_true):
            return False
    return True


def _rand_mat(rows, R, rng, half_prob=0.0):
    M = np.zeros((rows, R), dtype=np.int64)
    for r in range(rows):
        for c in range(R):
            v = rng.choice([0, 1, -1])
            if half_prob > 0.0 and rng.random() < half_prob:
                v = v * 0.5
            M[r, c] = v
    return M


def _search_rank(R, T, rng, budget):
    n, m, p = DIMENSIONS
    rowsU, rowsV, rowsW = n * m, m * p, p * n
    # Start with a structured random init: each column picks a random (i,j,k)
    while time.time() < budget:
        U = np.zeros((rowsU, R), dtype=np.int64)
        V = np.zeros((rowsV, R), dtype=np.int64)
        W = np.zeros((rowsW, R), dtype=np.int64)
        for c in range(R):
            i = rng.randrange(n)
            j = rng.randrange(m)
            k = rng.randrange(p)
            U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i, c] = 1
        # Random perturbations
        for _ in range(40):
            r = rng.randrange(R)
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                rr = rng.randrange(rowsU)
                U[rr, r] = rng.choice([0, 1, -1])
            elif mat == 1:
                rr = rng.randrange(rowsV)
                V[rr, r] = rng.choice([0, 1, -1])
            else:
                rr = rng.randrange(rowsW)
                W[rr, r] = rng.choice([0, 1, -1])
        if _verify(U, V, W, T, rng, trials=2):
            return U, V, W
    return None


def _to_list(M):
    out = []
    for r in range(M.shape[0]):
        row = []
        for c in range(M.shape[1]):
            v = M[r, c]
            if isinstance(v, float) and v * 2 == int(v * 2):
                v = int(v)
            row.append(v)
        out.append(row)
    return out


def construct():
    global _DEADLINE, _BEST, _BEST_RANK
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140.0
    if _BEST is not None:
        return _BEST

    rng = random.Random(12345)
    T = _build_tensor()
    n, m, p = DIMENSIONS
    schoolbook = n * m * p

    # Try ranks in increasing order; budget per rank shrinks as we go.
    start = 49
    # Total budget ~135s. Allocate more to lower ranks.
    ranks = list(range(start, schoolbook + 1))
    # We'll try a handful of ranks, spending time proportionally.
    # Give each rank a time slice; lower ranks get more.
    total_time = _DEADLINE - time.time() - 5.0  # 5s safety
    if total_time <= 0:
        total_time = 1.0
    # Weight: lower ranks get more time.
    weights = [1.0 / (r - start + 1) for r in ranks]
    wsum = sum(weights)
    slices = [(total_time * w) / wsum for w in weights]

    for idx, R in enumerate(ranks):
        if time.time() >= _DEADLINE - 2.0:
            break
        budget = min(time.time() + slices[idx], _DEADLINE - 2.0)
        result = _search_rank(R, T, rng, budget)
        if result is not None:
            U, V, W = result
            _BEST = {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
            _BEST_RANK = R
            return _BEST

    # Fallback: schoolbook
    rank = schoolbook
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
    _BEST = {"U": U, "V": V, "W": W}
    _BEST_RANK = rank
    return _BEST
# EVOLVE-BLOCK-END
