"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import fractions
from fractions import Fraction as Fr
import numpy as np

DIMENSIONS = [3, 3, 4]

_DEADLINE = time.time() + 150.0
_MARGIN = 3.0


def _matmul334():
    """Return the multiplication tensor T indexed as T[i*m + j, j*p + k, k*n + i]
    == delta_{ik}.  Here n=3, m=3, p=4."""
    n, m, p = DIMENSIONS
    T = [[[0.0] * (p * n) for _ in range(m * p)] for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j][j * p + k][k * n + i] = 1.0
    return T


def _to_fr(x):
    return Fr(str(x))


def _solve_linear(A, b):
    """Solve A x = b over the rationals.  A: list of list of Fr, b: list of Fr.
    Return x (list of Fr) or None if inconsistent (and None if singular-consistent
    would be underdetermined -> we only ever feed full-rank square systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = -1
        for r in range(col, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv < 0:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pc = M[col][col]
        for c in range(col, n + 1):
            M[col][c] = M[col][c] / pc
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                for c in range(col, n + 1):
                    M[r][c] = M[r][c] - f * M[col][c]
    return [M[i][n] for i in range(n)]


def _decompose_at_rank(R, T, tries, deadline):
    """Try to find a rank-R decomposition of tensor T (size a x b x c).
    Returns (U,V,W) as lists of lists of Fr, or None."""
    a, b, c = len(T), len(T[0]), len(T[0][0])
    Tfr = [[[_to_fr(T[i][j][k]) for k in range(c)] for j in range(b)] for i in range(a)]
    rng = random.Random(1000003 + R)
    for _ in range(tries):
        if time.time() > deadline:
            return None
        try:
            xs = [rng.uniform(-3, 3) for _ in range(a)]
            ys = [rng.uniform(-3, 3) for _ in range(b)]
            zs = [rng.uniform(-3, 3) for _ in range(c)]
            # linear map A: R^R -> R^a  (U from U*Vt*Wt)
            A = [[Fr(0)] * R for _ in range(a)]
            for i in range(a):
                for k in range(R):
                    s = Fr(0)
                    for j in range(b):
                        s += Tfr[i][j][k] * Fr(str(ys[j])) * Fr(str(zs[k]))
                    A[i][k] = s
            # linear map B: R^R -> R^b  (V from V*Ut*Wt)
            B = [[Fr(0)] * R for _ in range(b)]
            for j in range(b):
                for k in range(R):
                    s = Fr(0)
                    for i in range(a):
                        s += Tfr[i][j][k] * Fr(str(xs[i])) * Fr(str(zs[k]))
                    B[j][k] = s
            # linear map C: R^R -> R^c  (W from W*Ut*Vt)
            C = [[Fr(0)] * R for _ in range(c)]
            for k in range(c):
                for kk in range(R):
                    s = Fr(0)
                    for i in range(a):
                        for j in range(b):
                            s += Tfr[i][j][k] * Fr(str(xs[i])) * Fr(str(ys[j]))
                    C[k][kk] = s
            U = _solve_linear(A, [Fr(str(xs[i])) for i in range(a)])
            if U is None:
                continue
            V = _solve_linear(B, [Fr(str(ys[j])) for j in range(b)])
            if V is None:
                continue
            W = _solve_linear(C, [Fr(str(zs[k])) for k in range(c)])
            if W is None:
                continue
            # verify all tensor equations
            ok = True
            for i in range(a):
                rowok = True
                for j in range(b):
                    for k in range(c):
                        s = Fr(0)
                        for kk in range(R):
                            s += U[i][kk] * V[j][kk] * W[k][kk]
                        if s != Tfr[i][j][k]:
                            rowok = False
                            break
                    if not rowok:
                        break
                if not rowok:
                    ok = False
                    break
                if not ok:
                    break
            if ok:
                return U, V, W
        except Exception:
            continue
    return None


def _to_lists(U, V, W):
    a = len(U)
    b = len(V)
    c = len(W)
    R = len(U[0])
    return (
        [[float(U[i][k]) for k in range(R)] for i in range(a)],
        [[float(V[j][k]) for k in range(R)] for j in range(b)],
        [[float(W[k][k2]) for k2 in range(R)] for k in range(c)],
    )


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return U, V, W


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    a, b, c = n * m, m * p, p * n
    R = len(U[0])
    for i in range(a):
        for j in range(b):
            for k in range(c):
                s = 0.0
                for kk in range(R):
                    s += U[i][kk] * V[j][kk] * W[k][kk]
                if abs(s) > 1e-8:
                    return False
    return True


# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 150.0 - _MARGIN

    T = _matmul334()
    a, b, c = 3 * 3, 3 * 4, 4 * 3

    best_rank = 36
    U, V, W = _schoolbook()

    # Search strictly lower ranks, larger ranks first (easier to find if at all).
    # For a x b x c tensor, the rank-R equations are a+b+c unknowns (per factor
    # R columns) with a*b*c equations; keep tries modest and bounded by deadline.
    for R in range(35, 22, -1):
        if time.time() > deadline:
            break
        # scale tries: more for larger R, shrink as deadline approaches
        remaining = max(0.0, deadline - time.time())
        if remaining < 2.0:
            break
        base_tries = 120
        tries = max(20, int(base_tries * remaining / 150.0))
        res = _decompose_at_rank(R, T, tries, deadline)
        if res is not None:
            U2, V2, W2 = _to_lists(res[0], res[1], res[2])
            if _verify(U2, V2, W2):
                U, V, W = U2, V2, W2
                best_rank = R
                # found a valid lower rank; try to go even lower
                continue
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
