"""Matrix multiplication 2x4 by 4x8: rank-minimal exact decomposition.

Entry point: construct() -> {'U','V','W'}
Shapes: U (n*m, rank) = (8, r), V (m*p, rank) = (32, r), W (p*n, rank) = (16, r)
Trace convention: W[k*n + i, c].
"""
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
DEADLINE = 120.0          # evaluator wall-clock limit
SAFETY_MARGIN = 15.0      # keep a margin; stop searching well before the limit
_START = time.time()


def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START)


# ---------------------------------------------------------------------------
# Rank-56 baseline: eight independent 2x2 Strassen blocks.
#
# Group the 4x4 middle into two 2x2 blocks along the m=4 and p=8 axes.
# Block (a,b) with a in {0,1} (row pair of A), b in {0,1} (col pair of B):
#   A[:, 2a:2a+2] (2x2)  x  B[2a:2a+2, 2b:2b+2] (2x2)  ->  C[2a:2a+2, 2b:2b+2]
# Each uses 7 Strassen products.
# ---------------------------------------------------------------------------
_STRASSEN = [
    (0, 0, 0, 0, 0, 0),
    (1, 1, 0, 1, 0, 1),
    (0, 1, 1, 1, 1, 0),
    (1, 0, 0, 0, 1, 1),
    (0, 0, 1, 1, 0, 0),
    (1, 1, 0, 0, 0, 1),
    (0, 1, 1, 0, 1, 0),
]
_STRASSEN_SIGNS = [(1, 1, -1), (1, 1, 1), (1, 1, 1), (1, 1, -1),
                   (1, -1, 1), (1, 1, -1), (1, -1, 1)]


def _build_rank56():
    n, m, p = DIMENSIONS
    U = [[Fraction(0)] * 56 for _ in range(n * m)]   # 8 x 56
    V = [[Fraction(0)] * 56 for _ in range(m * p)]   # 32 x 56
    W = [[Fraction(0)] * 56 for _ in range(p * n)]   # 16 x 56
    col = 0
    for a in range(2):            # row-pair of A (and matching middle row-pair)
        for b in range(2):        # col-pair of B
            for idx, (ra, ca, rb, cb, cr, cc) in enumerate(_STRASSEN):
                sa, sb, sc = _STRASSEN_SIGNS[idx]
                A_idx = (2 * a + ra) * m + (2 * a + ca)
                V_idx = (2 * a + rb) * p + (4 * b + cb)
                W_idx = (4 * b + cc) * n + (2 * a + cr)
                U[A_idx][col] = Fraction(sa)
                V[V_idx][col] = Fraction(sb)
                W[W_idx][col] = Fraction(sc)
                col += 1
    return U, V, W


def _verify_clean(U, V, W, n, m, p, trials=6):
    """Exact rational verification on random integer matrices."""
    import random
    rng = random.Random(999)
    rank = len(U[0])
    for _ in range(trials):
        A = [[Fraction(rng.randint(-3, 3)) for _ in range(m)] for _ in range(n)]
        B = [[Fraction(rng.randint(-3, 3)) for _ in range(p)] for _ in range(m)]
        Cref = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    s += A[i][j] * B[j][k]
                Cref[i][k] = s
        P = [Fraction(0) for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += A[j][c] * V[j * p + k][c]
                P[j * p + k] = s
        C = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for j in range(m):
                    s += P[j * p + k] * W[k * n + i][j]
                C[i][k] = s
        for i in range(n):
            for k in range(p):
                if C[i][k] != Cref[i][k]:
                    return False
    return True


def _to_int_or_half(mat):
    """Convert a Fraction matrix to int/half-integer list-of-lists."""
    out = []
    for row in mat:
        r = []
        for v in row:
            f = Fraction(v)
            if f.denominator == 1:
                r.append(f.numerator)
            else:
                r.append(float(f))
        out.append(r)
    return out


def _try_drop_one(U, V, W, n, m, p):
    """Try to find a rank-(r-1) exact decomposition by zeroing one column and
    re-checking feasibility. Return (Ured,Vred,Wred) or None."""
    import random
    rank = len(U[0])
    rng = random.Random(7)
    candidates = list(range(rank))
    rng.shuffle(candidates)
    for c0 in candidates:
        if _time_left() < 5:
            return None
        keep = [c for c in range(rank) if c != c0]
        Ured = [[U[i][c] for c in keep] for i in range(len(U))]
        Vred = [[V[i][c] for c in keep] for i in range(len(V))]
        Wred = [[W[i][c] for c in keep] for i in range(len(W))]
        if _verify_clean(Ured, Vred, Wred, n, m, p, trials=3):
            return Ured, Vred, Wred
    return None


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank56()

    if not _verify_clean(U, V, W, n, m, p, trials=4):
        raise RuntimeError("rank-56 baseline failed verification")

    best = (U, V, W)

    if _time_left() > 30:
        cand = _try_drop_one(U, V, W, n, m, p)
        if cand is not None and _verify_clean(cand[0], cand[1], cand[2], n, m, p, trials=6):
            best = cand

    Ub, Vb, Wb = best
    return {
        "U": _to_int_or_half(Ub),
        "V": _to_int_or_half(Vb),
        "W": _to_int_or_half(Wb),
    }
