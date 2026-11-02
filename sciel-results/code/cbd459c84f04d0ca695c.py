"""Rank-minimal search for 5x5 matrix multiplication tensor decomposition."""
import time
import fractions
import itertools
import random

DIMENSIONS = [5, 5, 5]
N = M = P = 5

# Deadline guard: leave a safety margin before the 120s evaluator limit.
START = time.time()
DEADLINE = START + 112.0   # 8s safety margin


def _time_left():
    return DEADLINE - time.time()


def _build_schoolbook(n=N, m=M, p=P):
    """Rank n*m*p schoolbook factor matrices as lists of Fraction."""
    rank = n * m * p
    U = [[fractions.Fraction(0)] * rank for _ in range(n * m)]
    V = [[fractions.Fraction(0)] * rank for _ in range(m * p)]
    W = [[fractions.Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = fractions.Fraction(1)
                V[j * p + k][col] = fractions.Fraction(1)
                W[k * n + i][col] = fractions.Fraction(1)
                col += 1
    return U, V, W, rank


def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    equals 1 if i==k else 0.  Returns True/False (exact rational)."""
    rank = len(U[0]) if U else 0
    # Build V and W as lookup dicts keyed by (row, col) for speed, or just
    # iterate. We'll precompute for each term the product and accumulate.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = fractions.Fraction(1) if i == k else fractions.Fraction(0)
                acc = fractions.Fraction(0)
                urow = i * m + j
                vrow = j * p + k
                wrow = k * n + i
                for r in range(rank):
                    uv = U[urow][r] * V[vrow][r]
                    if uv != 0:
                        acc += uv * W[wrow][r]
                if acc != target:
                    return False
    return True


def _rank_of_columns(vectors):
    """Rank of a list of equal-length vectors (Fractions) via row reduction.
    vectors: list of lists.  Returns integer rank."""
    if not vectors:
        return 0
    mat = [list(v) for v in vectors]
    rows = len(mat)
    cols = len(mat[0])
    r = 0
    for c in range(cols):
        # find pivot
        piv = None
        for i in range(r, rows):
            if mat[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        # normalize pivot row
        for cc in range(c, cols):
            mat[r][cc] = mat[r][cc] / pv
        # eliminate
        for i in range(rows):
            if i != r and mat[i][c] != 0:
                f = mat[i][c]
                for cc in range(c, cols):
                    mat[i][cc] = mat[i][cc] - f * mat[r][cc]
        r += 1
        if r == rows:
            break
    return r


def _try_eliminate(U, V, W, n=N, m=M, p=P):
    """Attempt to remove one redundant term.  Returns (U,V,W,removed) where
    removed is True if a term was eliminated."""
    rank = len(U[0])
    # For each term r, test whether W[:,r] is in span of the other W columns.
    # We test via rank of W matrix with and without column r.
    base_rank = _rank_of_columns([W[i] for i in range(len(W))])
    for r in range(rank):
        if _time_left() < 6.0:
            return U, V, W, False
        cols = [W[i][:r] + W[i][r + 1:] for i in range(len(W))]
        if _rank_of_columns(cols) == base_rank:
            # column r is dependent -> try to remove term r
            newU = [row[:r] + row[r + 1:] for row in U]
            newV = [row[:r] + row[r + 1:] for row in V]
            newW = [row[:r] + row[r + 1:] for row in W]
            if _verify(newU, newV, newW, n, m, p):
                return newU, newV, newW, True
    return U, V, W, False


def _try_small_perturb(U, V, W, n=N, m=M, p=P):
    """Try small integer perturbations to a random term's entries to enable
    further reductions.  Returns (U,V,W,changed)."""
    rank = len(U[0])
    # pick a random term
    r = random.randrange(rank)
    # pick which matrix to perturb
    mat = random.choice(['U', 'V', 'W'])
    if mat == 'U':
        row = random.randrange(len(U))
        base = list(U[row])
    elif mat == 'V':
        row = random.randrange(len(V))
        base = list(V[row])
    else:
        row = random.randrange(len(W))
        base = list(W[row])
    # perturb entry r by a small integer
    for delta in (1, -1, 2, -2):
        newbase = list(base)
        newbase[r] = newbase[r] + fractions.Fraction(delta)
        if mat == 'U':
            newU = [row[:] for row in U]; newU[row] = newbase
            cand = (newU, V, W)
        elif mat == 'V':
            newV = [row[:] for row in V]; newV[row] = newbase
            cand = (U, newV, W)
        else:
            newW = [row[:] for row in W]; newW[row] = newbase
            cand = (U, V, newW)
        if _verify(cand[0], cand[1], cand[2], n, m, p):
            return cand[0], cand[1], cand[2], True
    return U, V, W, False


def construct():
    t0 = time.time()
    # Start from schoolbook (always valid, rank 125).
    U, V, W, rank = _build_schoolbook()
    best = (U, V, W, rank)

    # Iteratively eliminate redundant terms.
    progress = True
    while progress and _time_left() > 12.0:
        progress = False
        # Try elimination a few times per outer loop.
        for _ in range(40):
            if _time_left() < 12.0:
                break
            U, V, W, removed = _try_eliminate(U, V, W)
            if removed:
                rank = len(U[0])
                progress = True
                if rank < best[3]:
                    best = (U, V, W, rank)
                break  # re-scan after a removal
        # Occasionally perturb to escape local minima.
        if _time_left() > 20.0 and random.random() < 0.5:
            U2, V2, W2, changed = _try_small_perturb(U, V, W)
            if changed:
                U, V, W = U2, V2, W2
                rank = len(U[0])
                # re-eliminate after perturbation
                progress = True

    # Ensure best is valid (it always is, but be safe).
    bU, bV, bW, brank = best
    if not _verify(bU, bV, bW):
        # fall back to a fresh schoolbook
        bU, bV, bW, brank = _build_schoolbook()

    # Convert Fractions to int/half-integer (float) for the interface.
    def _to_num(f):
        v = float(f)
        return v

    Uout = [[_to_num(x) for x in row] for row in bU]
    Vout = [[_to_num(x) for x in row] for row in bV]
    Wout = [[_to_num(x) for x in row] for row in bW]
    return {"U": Uout, "V": Vout, "W": Wout}


# Self-test on import (cheap, keeps construct reachable).
if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
