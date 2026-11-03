"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE_SECONDS = 150
_START_TIME = time.time()
_RNG_SEED = 12345


def _elapsed():
    return time.time() - _START_TIME


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Exact check: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all."""
    rank = len(U[0])
    Um = np.array(U, dtype=float).reshape(n, m, rank)
    Vm = np.array(V, dtype=float).reshape(m, p, rank)
    Wm = np.array(W, dtype=float).reshape(p, n, rank)
    # vectorized over r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('r,r,r->', Um[i, j], Vm[j, k], Wm[k, i])
                if abs(float(val) - 1.0) > tol:
                    return False
    return True


def _to_lists(arr2d):
    return [[int(round(x * 2)) // 2 if abs(x - round(x)) < 1e-9 else x
             for x in row] for row in arr2d]


def _schoolbook(n, m, p):
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


def _round_to_half(x):
    return round(x * 2) / 2.0


def _search_structured(n, m, p, target_ranks, attempts_per_rank):
    """Search for a verified rank below the schoolbook rank by attempting random
    half-integer factorizations guided by the tensor structure. Only returns a
    candidate that passes exact verification; otherwise None."""
    rng = random.Random(_RNG_SEED)
    for r in target_ranks:
        for _ in range(attempts_per_rank):
            if _elapsed() > _DEADLINE_SECONDS - 20:
                return None
            # Initialize from schoolbook prefix then randomly perturb and re-solve
            # a small number of columns to try to merge contributions.
            base_rank = n * m * p
            Um = np.zeros((n, m, r))
            Vm = np.zeros((m, p, r))
            Wm = np.zeros((p, n, r))
            # seed columns with structured 1/2 patterns
            for col in range(r):
                # random sparse 1/0/+-1 pattern across the three factors
                for i in range(n):
                    for j in range(m):
                        if rng.random() < 0.4:
                            Um[i, j, col] = rng.choice([-1, 1])
                for j in range(m):
                    for k in range(p):
                        if rng.random() < 0.4:
                            Vm[j, k, col] = rng.choice([-1, 1])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.4:
                            Wm[k, i, col] = rng.choice([-1, 1])
            # Round to half-integers
            Um = np.round(Um * 2) / 2
            Vm = np.round(Vm * 2) / 2
            Wm = np.round(Wm * 2) / 2
            U = np.repeat(Um.reshape(n * m, r), 1, axis=1)  # no-op reshape sanity
            Ulist = [[_round_to_half(v) for v in Um[i * m + j]] for i in range(n) for j in range(m)]
            Vlist = [[_round_to_half(v) for v in Vm[j * p + k]] for j in range(m) for k in range(p)]
            Wlist = [[_round_to_half(v) for v in Wm[k * n + i]] for k in range(p) for i in range(n)]
            if _verify(Ulist, Vlist, Wlist, n, m, p):
                return {"U": Ulist, "V": Vlist, "W": Wlist}
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    school_rank = len(best["U"][0])

    # Attempt a verified improvement; only accept on exact verification.
    target_ranks = list(range(max(1, school_rank - 8), school_rank))
    found = _search_structured(n, m, p, target_ranks, attempts_per_rank=40)
    if found is not None and len(found["U"][0]) < school_rank:
        best = found
    return best
# EVOLVE-BLOCK-END
