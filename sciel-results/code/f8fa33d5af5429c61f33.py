"""Combined program for matrix multiplication tensor decomposition 4x4 * 4x5.

Strategy:
- Start from the schoolbook rank-80 baseline.
- Search lower ranks (from a low bound up) for an exact decomposition.
- Use a fast numpy float pre-filter followed by exact Fraction verification.
- Mix random sampling with Strassen-structured seeds and local search.
- Hard deadline guard: return the best valid result before the 150s wall-clock limit.
"""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

# Precompute the index mapping for each (i, j, k) triple.
# U index: i*m + j ; V index: j*p + k ; W index: k*n + i
_TRIPLES = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
_UIDX = [i * m + j for (i, j, k) in _TRIPLES]
_VIDX = [j * p + k for (i, j, k) in _TRIPLES]
_WIDX = [k * n + i for (i, j, k) in _TRIPLES]
_NT = len(_TRIPLES)


def _schoolbook():
    """Rank n*m*p decomposition: one term per (i,j,k)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for (i, j, k) in _TRIPLES:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2.

    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    # M6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    return U, V, W


def _verify_float(U, V, W, rank, tol=1e-7):
    """Fast numpy pre-filter. Returns True if the tensor is close to all-ones."""
    U_np = np.asarray(U, dtype=np.float64)
    V_np = np.asarray(V, dtype=np.float64)
    W_np = np.asarray(W, dtype=np.float64)
    # Gather rows for each triple.
    Ur = U_np[_UIDX]          # (NT, rank)
    Vr = V_np[_VIDX]          # (NT, rank)
    Wr = W_np[_WIDX]          # (NT, rank)
    # T[t] = sum_c Ur[t,c] * Vr[t,c] * Wr[t,c]
    vals = (Ur * Vr * Wr).sum(axis=1)
    return bool(np.allclose(vals, 1.0, atol=tol))


def _verify_exact(U, V, W, rank):
    """Exact Fraction verification. Returns True iff every triple sums to exactly 1."""
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for t in range(_NT):
        ui = Uf[_UIDX[t]]
        vj = Vf[_VIDX[t]]
        wk = Wf[_WIDX[t]]
        total = Fraction(0)
        for c in range(rank):
            total += ui[c] * vj[c] * wk[c]
        if total != 1:
            return False
    return True


def _is_valid(U, V, W, rank):
    """Two-stage validation: fast float pre-filter then exact check."""
    try:
        if not _verify_float(U, V, W, rank):
            return False
    except Exception:
        return False
    try:
        return _verify_exact(U, V, W, rank)
    except Exception:
        return False


def _random_candidate(rank, rng, values):
    U = [[rng.choice(values) for _ in range(rank)] for _ in range(NM)]
    V = [[rng.choice(values) for _ in range(rank)] for _ in range(MP)]
    W = [[rng.choice(values) for _ in range(rank)] for _ in range(PN)]
    return U, V, W


def _sparse_candidate(rank, rng, values):
    """Start sparse then sprinkle non-zeros (Strassen-like sparsity)."""
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    nz = max(3, int(rank * 0.5))
    for _ in range(nz):
        u_row = rng.randrange(NM)
        v_row = rng.randrange(MP)
        w_row = rng.randrange(PN)
        col = rng.randrange(rank)
        U[u_row][col] = rng.choice([1, -1])
        V[v_row][col] = rng.choice([1, -1])
        W[w_row][col] = rng.choice([1, -1])
    # A few random extra fills to help reach the target sum.
    for _ in range(int(rank * 0.3)):
        mat = rng.choice([U, V, W])
        rows = NM if mat is U else (MP if mat is V else PN)
        mat[rng.randrange(rows)][rng.randrange(rank)] = rng.choice(values)
    return U, V, W


def _local_search(U, V, W, rank, rng, values, iters, deadline):
    """Mutate single entries; return (U,V,W) if valid, else None."""
    mats = [U, V, W]
    rows = [NM, MP, PN]
    for _ in range(iters):
        if time.time() > deadline:
            break
        mi = rng.randrange(3)
        mat = mats[mi]
        mat[rng.randrange(rows[mi])][rng.randrange(rank)] = rng.choice(values)
        if _is_valid(U, V, W, rank):
            return U, V, W
    return None


def _try_rank(rank, budget, seed, deadline):
    """Spend up to `budget` seconds trying to find a valid decomposition at `rank`.
    Returns (U,V,W) or None.
    """
    if time.time() >= deadline:
        return None
    rng = random.Random(seed)
    values = [-1, 0, 1]
    t0 = time.time()

    # Phase A: random full candidates (fast float pre-filter keeps this cheap).
    while time.time() < min(t0 + budget, deadline):
        U, V, W = _random_candidate(rank, rng, values)
        if _is_valid(U, V, W, rank):
            return U, V, W

    # Phase B: sparse seeds + local search (structured).
    if time.time() < deadline:
        for attempt in range(6):
            if time.time() > deadline:
                break
            U, V, W = _sparse_candidate(rank, rng, values)
            res = _local_search(U, V, W, rank, rng, values, iters=120, deadline=deadline)
            if res is not None:
                return res

    # Phase C: half-integer values, a few random tries.
    if time.time() < deadline:
        values2 = [-1, -0.5, 0, 0.5, 1]
        for _ in range(40):
            if time.time() > deadline:
                break
            U, V, W = _random_candidate(rank, rng, values2)
            if _is_valid(U, V, W, rank):
                return U, V, W

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock limit

    # Baseline: schoolbook rank 80 (always valid).
    best = _schoolbook()
    best_rank = n * m * p

    # Search lower ranks first (most likely to beat baseline).
    # Theoretical lower bound is >= max(n*m, m*p, p*n) = 20.
    min_rank = 16
    max_rank = best_rank - 1  # 79

    # Allocate the search budget across ranks, weighting low ranks more heavily.
    # We iterate ascending so we try the most promising (lowest) ranks first.
    total_budget = deadline - start_time - 3.0
    total_budget = max(total_budget, 1.0)

    # Give each rank a share; lower ranks get slightly more.
    num_ranks = max_rank - min_rank + 1
    # Weights: descending so rank=min_rank gets the most.
    weights = [max(1, (max_rank - r + 1)) for r in range(min_rank, max_rank + 1)]
    wsum = float(sum(weights))

    seed = 12345
    for r in range(min_rank, max_rank + 1):
        if time.time() >= deadline:
            break
        idx = r - min_rank
        share = total_budget * (weights[idx] / wsum)
        share = max(share, 0.15)  # at least a little per rank
        res = _try_rank(r, share, seed, deadline)
        seed += 101
        if res is not None:
            best_rank = r
            best = {"U": res[0], "V": res[1], "W": res[2]}
            # Found a valid lower rank; keep going to try even lower.

    return best
