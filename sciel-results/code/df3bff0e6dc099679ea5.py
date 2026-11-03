"""Matrix multiplication 6x6x6: search for a rank-minimal exact tensor decomposition."""
import random
import time
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

# Deadline guard: evaluator kills at 150s; leave a safety margin.
_START = time.time()
_DEADLINE = 150.0 - 12.0  # 138s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START)

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
    return U, V, W

def _validate(U, V, W, n, m, p):
    # Exact integer check of the matrix-multiplication tensor.
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if s != 1:
                    return False
    # Orthogonality: distinct (i,j,k) must give 0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                for k2 in range(p):
                    if k2 == k:
                        continue
                    wk = k2 * n + i
                    s = 0
                    for c in range(rank):
                        s += U[ui][c] * V[vj][c] * W[wk][c]
                    if s != 0:
                        return False
    return True

def _rank_of(U):
    return len(U[0])

def _try_patch(U, V, W, n, m, p, rng):
    """Attempt to reduce rank by removing a redundant term and re-adding a cheaper one.
    Returns a new (U,V,W) if it is valid and lower rank, else None."""
    rank = _rank_of(U)
    if rank <= 1:
        return None
    # Try dropping one random term (set its column to zero) then check if still valid.
    # If still valid, the term was redundant -> we can remove it.
    for attempt in range(3):
        c = rng.randrange(rank)
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        for r in range(len(U2)):
            U2[r][c] = 0
        for r in range(len(V2)):
            V2[r][c] = 0
        for r in range(len(W2)):
            W2[r][c] = 0
        if _validate(U2, V2, W2, n, m, p):
            # Term c was redundant; drop it.
            U3 = [row[:c] + row[c + 1:] for row in U2]
            V3 = [row[:c] + row[c + 1:] for row in V2]
            W3 = [row[:c] + row[c + 1:] for row in W2]
            return U3, V3, W3
    return None

def _random_sparse_decomp(n, m, p, rank, rng, entries):
    """Build a random sparse decomposition of the given rank and validate."""
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        # choose one support per factor
        ui = rng.randrange(n * m)
        vj = rng.randrange(m * p)
        wk = rng.randrange(p * n)
        U[ui][c] = rng.choice(entries)
        V[vj][c] = rng.choice(entries)
        W[wk][c] = rng.choice(entries)
    if _validate(U, V, W, n, m, p):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = _rank_of(U)
    best = (U, V, W)
    rng = random.Random(12345)

    # Phase 1: try to prune redundant terms from schoolbook.
    for _ in range(2000):
        if _time_left() < 1.0:
            break
        res = _try_patch(U, V, W, n, m, p, rng)
        if res is not None:
            U, V, W = res
            if _rank_of(U) < best_rank:
                best_rank = _rank_of(U)
                best = (U, V, W)

    # Phase 2: random sparse searches for lower ranks (bounded by deadline).
    entries = [1, -1, 2, -2, 1 / 2, -1 / 2]
    for target in range(best_rank - 1, max(1, best_rank - 40), -1):
        if _time_left() < 1.0:
            break
        iters = 0
        while _time_left() > 0.5:
            res = _random_sparse_decomp(n, m, p, target, rng, entries)
            if res is not None:
                U, V, W = res
                best_rank = target
                best = (U, V, W)
                break
            iters += 1
            if iters > 2000:
                break

    U, V, W = best
    return {"U": U, "V": V, "W": W}
