"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import log, exp

def _softmin_neg(sum_squares: float, scale: float) -> float:
    # Smooth surrogate for min over 27 entries of |residual|, using -sum exp(-scale*|r|).
    # (log-sum-exp style; avoids branching on a nonsmooth min.)
    if sum_squares <= 0:
        return 0.0
    return -exp(-scale) * sum_squares  # not exact soft-min but monotone surrogate

def _verify(U, V, W) -> bool:
    n = m = p = 3
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    u = U[i*m + j][c]
                    v = V[j*p + k][c]
                    w = W[k*n + i][c]
                    if u == 0 or v == 0 or w == 0:
                        continue
                    s += Fraction(u) * Fraction(v) * Fraction(w)
                if s != 1:
                    return False
    return True

def _residuals(U, V, W):
    n = m = p = 3
    rank = len(U[0])
    # residuals indexed by (i,j,k) -> sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] - 1
    res = [[0.0]*p for j in range(m) for i in range(n)]  # flattened 27
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i*m + j]
                vj = V[j*p + k]
                wk = W[k*n + i]
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                res[idx] = s - 1.0
                idx += 1
    return res

def _objective(U, V, W, scale: float) -> float:
    res = _residuals(U, V, W)
    sum_squares = 0.0
    for r in res:
        sum_squares += r*r
    return _softmin_neg(sum_squares, scale)

def _random_sparse_init(rank, density, seed):
    rng = random.Random(seed)
    n = m = p = 3
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        # each column: pick a few non-zeros in each matrix
        for r in range(n*m):
            if rng.random() < density:
                U[r][c] = rng.choice([-1, 1])
        for r in range(m*p):
            if rng.random() < density:
                V[r][c] = rng.choice([-1, 1])
        for r in range(p*n):
            if rng.random() < density:
                W[r][c] = rng.choice([-1, 1])
    return U, V, W

def _exact_local_polish(U, V, W, max_iters, deadline):
    # Try exact single-coordinate updates: for a fixed (matrix, row, col),
    # solve the linear system over the 27 entries? That's nonlinear in that coord.
    # Instead do small exact local search: try flipping 0<->1 and 1<->-1 on a few coords.
    n = m = p = 3
    rank = len(U[0])
    # Precompute for speed: store as list of lists
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_obj = _objective(best_U, best_V, best_W, scale=8.0)
    if best_obj < 1e-12:
        return best_U, best_V, best_W
    # Simple hill-climb with exact verification occasionally
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # pick random coordinate
        mat = random.randint(0, 2)
        if mat == 0:
            rows = n*m
            M = best_U
        elif mat == 1:
            rows = m*p
            M = best_V
        else:
            rows = p*n
            M = best_W
        r = random.randrange(rows)
        c = random.randrange(rank)
        old = M[r][c]
        if old == 0:
            new = random.choice([-1, 1])
        else:
            new = 0 if random.random() < 0.5 else (-1 if old == 1 else 1)
        M[r][c] = new
        obj = _objective(M and (best_U if mat==0 else best_V if mat==1 else best_W),
                         best_V if mat==0 else (best_U if mat==1 else best_V),
                         best_W if mat==0 else (best_U if mat==1 else best_V),
                         scale=8.0)
        # Recompute properly with correct args
        if mat == 0:
            obj = _objective(best_U, best_V, best_W, scale=8.0)
        elif mat == 1:
            obj = _objective(best_U, best_V, best_W, scale=8.0)
        else:
            obj = _objective(best_U, best_V, best_W, scale=8.0)
        if obj < best_obj - 1e-12:
            best_obj = obj
            if best_obj < 1e-12:
                # exact verify
                if _verify(best_U, best_V, best_W):
                    return best_U, best_V, best_W
        else:
            M[r][c] = old
    return best_U, best_V, best_W

def _search_rank(r, time_budget, seed):
    n = m = p = 3
    deadline = time.time() + time_budget
    best = None
    best_obj = float('inf')
    # Try several random starts
    max_starts = 40
    for s in range(max_starts):
        if time.time() > deadline:
            break
        U, V, W = _random_sparse_init(r, density=0.35, seed=seed + s)
        obj = _objective(U, V, W, scale=8.0)
        if obj < best_obj:
            best_obj = obj
            best = (U, V, W)
        # Coordinate descent
        U, V, W = _exact_local_polish(U, V, W, max_iters=200, deadline=deadline)
        obj2 = _objective(U, V, W, scale=8.0)
        if obj2 < best_obj:
            best_obj = obj2
            best = (U, V, W)
        if best_obj < 1e-12:
            if _verify(U, V, W):
                return U, V, W
    if best is not None:
        U, V, W = best
        if _verify(U, V, W):
            return U, V, W
    return None

def _construct_schoolbook():
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

def construct():
    overall_deadline = time.time() + 140  # safety margin before 150s kill
    random.seed(12345)
    # Search from low rank upward
    for r in [20, 21, 22, 23]:
        if time.time() > overall_deadline - 5:
            break
        result = _search_rank(r, time_budget=30, seed=1000 + r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    # Fallback: schoolbook rank 27
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
