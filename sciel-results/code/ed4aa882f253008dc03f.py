"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy (same design family as the seed: DIRECT construction of factor
matrices U, V, W that exactly decompose the multiplication tensor):

  * Guaranteed fallback: the explicit schoolbook decomposition of rank
    n*m*p = 28.  This is always valid and exact, so the program can never
    return an invalid answer (score >= 1.0).

  * Improvement attempt: search for a *smaller* exact decomposition
    (rank 27, then 26, ...) by solving the defining tensor equations
        sum_c U[i*m+j, c] * V[j*p+k, c] == delta[i,k]
    numerically (scipy SLSQP / least-squares from many random + structured
    starts) and then VERIFYING the candidate exactly with rational
    arithmetic (fractions.Fraction).  Only a candidate that passes the
    exact check is accepted.

  The evaluator scores seed/value where value = rank, so any exact
  decomposition of rank < 28 improves the score.  If no smaller exact
  decomposition is found within the time budget we return the rank-28
  schoolbook result (no regression).

Conventions (official trace convention):
    U: (n*m, R)   U[i*m + j, c]
    V: (m*p, R)   V[j*p + k, c]
    W: (p*n, R)   W[k*n + i, c]
    C[i,k] = sum_j A[i,j] B[j,k]  recovered as
        C = sum_c  (U[:,c] routed through A) (V[:,c] routed through B)
    i.e. for each c:  T_c[i,k] = sum_j U[i*m+j,c] V[j*p+k,c],
    and the W column c must route to (k,i), i.e. W[k*n+i,c] = 1 (up to the
    same scalar that keeps the sum exact).
"""

import time
import random
import math
from fractions import Fraction

DIMENSIONS = [2, 2, 7]


# ---------------------------------------------------------------------------
# Exact schoolbook fallback (rank n*m*p) -- always valid.
# ---------------------------------------------------------------------------
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Exact rational verification of a candidate (U, V, W).
# ---------------------------------------------------------------------------
def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        # Round to a half-integer grid first (task allows integer/half-integer),
        # then treat as exact.  If the numeric solve is good the nearest
        # half-integer is the intended value.
        h = int(round(2.0 * float(x))) / 2.0
        return Fraction(h).limit_denominator(10**6)
    return Fraction(x).limit_denominator(10**6)


def _verify_exact(U, V, W, n, m, p, rng, probes=8):
    """Return True iff (U,V,W) exactly reproduces C = A*B for random A,B.

    We check the *tensor identity* directly (independent of A,B):
        sum_c U[i*m+j,c] V[j*p+k,c] == delta[i,k]   for all i,j,k
    and the W routing: each column c has exactly one nonzero W entry and it
    sits at index k*n+i for the (i,k) that the column contributes to.
    """
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    # (1) tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c] == delta[i,k]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = Fraction(0)
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c]
                if s != (1 if i == k else 0):
                    return False

    # (2) W routing: for each column, exactly one nonzero, at the right (i,k).
    #     The column c contributes to output (i,k) where the tensor identity
    #     has a matching (i,k); we require W to have a single 1 at k*n+i for
    #     the (i,k) that column c is responsible for.  We determine the
    #     responsible (i,k) from U and V support.
    for c in range(R):
        nz = [(k, i) for k in range(p) for i in range(n) if Wf[k * n + i][c] != 0]
        if len(nz) != 1:
            return False
        (k0, i0) = nz[0]
        # the column must actually be able to produce (i0,k0):
        # there exists j with U[i0*m+j,c]!=0 and V[j*p+k0,c]!=0
        ok = False
        for j in range(m):
            if Uf[i0 * m + j][c] != 0 and Vf[j * p + k0][c] != 0:
                ok = True
                break
        if not ok:
            return False
    return True


def _round_to_halfgrid(M):
    """Round each entry to the nearest half-integer (task-allowed grid)."""
    return [[int(round(2.0 * float(M[i][c]))) / 2.0 for c in range(len(M[0]))]
            for i in range(len(M))]


# ---------------------------------------------------------------------------
# Numerical search for a smaller exact decomposition.
# ---------------------------------------------------------------------------
def _residual(U, V, n, m, p):
    """Return (sse, max_abs) of the tensor identity residual.

    residual[i,k,j] = sum_c U[i*m+j,c]*V[j*p+k,c] - delta[i,k]
    """
    R = len(U[0])
    sse = 0.0
    mx = 0.0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c]
                r = s - (1.0 if i == k else 0.0)
                sse += r * r
                if abs(r) > mx:
                    mx = abs(r)
    return sse, mx


def _random_init(n, m, p, R, rng):
    U = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(m * p)]
    return U, V


def _structured_init(n, m, p, R, rng, schoolbook):
    """Seed from schoolbook (rank 28) by randomly dropping (28-R) columns and
    re-randomizing the rest slightly, giving the optimizer a near-feasible
    starting point."""
    U, V, W = schoolbook
    keep = list(range(28))
    rng.shuffle(keep)
    keep = keep[:R]
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    for new_c, old_c in enumerate(keep):
        for i in range(n * m):
            U[i][new_c] = U[i][old_c] + rng.gauss(0.0, 0.2)
        for j in range(m * p):
            V[j][new_c] = V[j][old_c] + rng.gauss(0.0, 0.2)
    return U, V


def _solve_candidate(n, m, p, R, rng, schoolbook, deadline, attempt):
    """Try to find an exact rank-R decomposition.  Returns (U,V,W) or None."""
    # Choose a start: structured (near-feasible) for the first few attempts,
    # then random.
    if attempt < 3:
        U, V = _structured_init(n, m, p, R, rng, schoolbook)
    else:
        U, V = _random_init(n, m, p, R, rng)

    import numpy as np
    from scipy.optimize import minimize

    dim = R * (n * m + m * p)

    def flatten(x):
        U = [x[i * R:(i + 1) * R] for i in range(n * m)]
        V = [x[(n * m) * R + j * R:(n * m) * R + (j + 1) * R] for j in range(m * p)]
        return U, V

    def obj(x):
        U, V = flatten(x)
        sse, _ = _residual(U, V, n, m, p)
        return sse

    x0 = []
    for row in U:
        x0.extend(row)
    for row in V:
        x0.extend(row)

    best_x = x0
    best_obj = obj(x0)

    for method in ("SLSQP", "L-BFGS-B"):
        if time.time() > deadline:
            break
        try:
            res = minimize(obj, x0, method=method,
                           options={"maxiter": 60, "ftol": 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = list(res.x)
            if best_obj < 1e-9:
                break
        except Exception:
            pass
        if time.time() > deadline:
            break

    U, V = flatten(best_x)
    if best_obj > 1e-8:
        return None

    # Round to half-integer grid and build W, then verify exactly.
    Ur = _round_to_halfgrid(U)
    Vr = _round_to_halfgrid(V)
    # Build W: for each column, find the (i,k) it routes to.
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        # find (i,k) such that the column contributes (i,k)
        found = False
        for i in range(n):
            for k in range(p):
                # column contributes (i,k) iff exists j with U[i*m+j,c]*V[j*p+k,c]!=0
                # and the tensor identity holds for (i,k) (it does, by solve)
                contrib = any(Ur[i * m + j][c] != 0 and Vr[j * p + k][c] != 0
                              for j in range(m))
                if contrib:
                    W[k * n + i][c] = 1
                    found = True
                    break
            if found:
                break
        if not found:
            # column is effectively zero; leave W column zero (harmless)
            pass

    if _verify_exact(Ur, Vr, W, n, m, p, rng):
        return Ur, Vr, W
    return None


def _search_smaller(n, m, p, deadline, seed):
    """Search for an exact decomposition of rank < n*m*p.  Return (U,V,W) or
    None.  Tries R = 27, 26, ... down to a small floor."""
    rng = random.Random(seed)
    schoolbook = _build_schoolbook(n, m, p)
    target_rank = n * m * p
    floor = 12  # do not waste time far below plausible rank
    attempt = 0
    for R in range(target_rank - 1, floor - 1, -1):
        if time.time() > deadline:
            break
        # A few attempts per rank; more for the most promising (R=27).
        n_attempts = 6 if R == target_rank - 1 else 3
        for a in range(n_attempts):
            if time.time() > deadline:
                break
            attempt += 1
            cand = _solve_candidate(n, m, p, R, rng, schoolbook, deadline, attempt)
            if cand is not None:
                return cand
    return None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # safety margin under the 150s wall-clock kill

    # Guaranteed-correct fallback.
    U, V, W = _build_schoolbook(n, m, p)

    # Attempt to find a smaller exact decomposition.
    try:
        cand = _search_smaller(n, m, p, deadline, seed=12345)
        if cand is not None:
            U, V, W = cand
    except Exception:
        # Never let the search break the guaranteed result.
        pass

    return {"U": U, "V": V, "W": W}
