"""Fresh design: iterative rank-reduction of the matrix-multiplication tensor
for 3x3 x 3x4, using exact rational arithmetic and a wall-clock deadline guard."""
import time
from fractions import Fraction
import random

DIMENSIONS = [3, 3, 4]

def _mul_tensor(n, m, p):
    """Return the (n*m*p)-dim vector of the MM tensor in (i,j,k) order."""
    # T[i,j,k] = 1 for all (i,j,k); the MM tensor is actually the identity-like
    # tensor T[i,j,k] = delta(i,j,k)? No.
    # The matrix multiplication tensor T has T[i,j,k] = 1 for every (i,j,k)
    # when we write (AB)[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is the all-ones tensor of shape (n,m,p).
    return [Fraction(1) for _ in range(n * m * p)]

def _decompose_all_ones(n, m, p):
    """Schoolbook: one rank-1 term per (i,j,k)."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = Fraction(1)
                V[j * p + k][c] = Fraction(1)
                W[k * n + i][c] = Fraction(1)
                c += 1
    return U, V, W

def _check(U, V, W, n, m, p):
    """Verify sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != Fraction(1):
                    return False
    return True

def _try_reduce(U, V, W, n, m, p, rng):
    """Attempt to reduce rank by 1 by finding a linear relation among terms.
    If term a can be expressed as a linear combination of the others in the
    tensor sense, drop it. We try a randomized elimination: pick two terms
    and try to merge them by adding a multiple of one to the other, then
    zeroing one out.
    """
    r = len(U[0])
    if r <= 1:
        return None

    # Try to find c1 != c2 such that the product vectors are linearly dependent
    # in the appropriate sense. A simpler heuristic: for each pair (a,b), try
    # to eliminate b by setting W_b = W_b - alpha * W_a for all alpha in a
    # small set, and check if that makes some component of some term zero
    # allowing removal. This is expensive; instead we use a direct linear
    # algebra approach on the tensor.

    # Build the tensor as a flat vector and the rank-1 contributions.
    N = n * m * p
    # contribution of column c: vec_c[i*m*p + j*p + k] = U[i,j,c]*V[j,k,c]*W[k,i,c]
    contribs = []
    for c in range(r):
        vec = [Fraction(0)] * N
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = (U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
        contribs.append(vec)

    # The sum of all contribs must equal the all-ones vector.
    # We want to see if some contrib is a linear combination of the rest.
    # Pick a random column to eliminate.
    for elim in range(r):
        # Try: can we find coefficients x_c (c != elim) such that
        # contrib_elim = sum_c x_c * contrib_c?
        # That's a linear system of N equations in (r-1) unknowns.
        # Since N=36 and r-1 < 36 typically, it's overdetermined.
        # Instead, we try a different reduction: find alpha such that
        # contrib_a + alpha * contrib_b has a zero in some "support" that
        # allows merging.
        pass

    # Simpler and effective: random linear combination elimination.
    # Pick two random columns a, b. For a random alpha, form new_b = b + alpha*a.
    # If for the new set, column a becomes redundant (its vector is in span of others),
    # drop it.
    for _ in range(200):
        a = rng.randrange(r)
        b = rng.randrange(r)
        if a == b:
            continue
        # alpha from a small rational set
        alpha = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                            Fraction(2), Fraction(-2), Fraction(1, 3), Fraction(-1, 3)])
        # new column b = old b + alpha * old a
        newU = [row[:] for row in U]
        newV = [row[:] for row in V]
        newW = [row[:] for row in W]
        for i in range(n * m):
            newU[i][b] = newU[i][b] + alpha * newU[i][a]
        for j in range(m * p):
            newV[j][b] = newV[j][b] + alpha * newV[j][a]
        for k in range(p * n):
            newW[k][b] = newW[k][b] + alpha * newW[k][a]
        # Now check if column a is zero (in the tensor sense) i.e. can be dropped.
        # Actually we changed b, not a. We want to make a zero.
        # Instead: set new_a = a - alpha * b won't help. Let's instead try to
        # zero out column a by subtracting a multiple of b from a.
        # Try: new_a = a + beta * b for random beta, then check if new_a's
        # tensor vector is zero.
        for _ in range(50):
            beta = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                               Fraction(2), Fraction(-2), Fraction(1, 3), Fraction(-1, 3),
                               Fraction(1, 4), Fraction(-1, 4)])
            tU = [row[:] for row in U]
            tV = [row[:] for row in V]
            tW = [row[:] for row in W]
            for i in range(n * m):
                tU[i][a] = tU[i][a] + beta * tU[i][b]
            for j in range(m * p):
                tV[j][a] = tV[j][a] + beta * tV[j][b]
            for k in range(p * n):
                tW[k][a] = tW[k][a] + beta * tW[k][b]
            # check if column a tensor vector is all zero
            zero = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = tU[i * m + j][a] * tV[j * p + k][a] * tW[k * n + i][a]
                        if val != Fraction(0):
                            zero = False
                            break
                    if not zero:
                        break
                if not zero:
                    break
            if zero:
                # drop column a
                U2 = [row[:a] + row[a + 1:] for row in tU]
                V2 = [row[:a] + row[a + 1:] for row in tV]
                W2 = [row[:a] + row[a + 1:] for row in tW]
                if _check(U2, V2, W2, n, m, p):
                    return U2, V2, W2
    return None

def construct():
    start = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    U, V, W = _decompose_all_ones(n, m, p)
    best = (U, V, W)
    best_rank = len(U[0])

    # Iteratively try to reduce rank.
    while len(U[0]) > 1:
        if time.time() - start > DEADLINE - MARGIN:
            break
        res = _try_reduce(U, V, W, n, m, p, rng)
        if res is not None:
            U, V, W = res
            if len(U[0]) < best_rank:
                best_rank = len(U[0])
                best = (U, V, W)
        else:
            # try a few more random restarts of the local search
            for _ in range(5):
                if time.time() - start > DEADLINE - MARGIN:
                    break
                res = _try_reduce(U, V, W, n, m, p, rng)
                if res is not None:
                    U, V, W = res
                    if len(U[0]) < best_rank:
                        best_rank = len(U[0])
                        best = (U, V, W)
                    break
            else:
                # stuck; break out
                break

    U, V, W = best
    # Convert Fractions to int or half-int (float) as required.
    def to_num(f):
        if f.denominator == 1:
            return int(f)
        else:
            return float(f)
    Uo = [[to_num(x) for x in row] for row in U]
    Vo = [[to_num(x) for x in row] for row in V]
    Wo = [[to_num(x) for x in row] for row in W]
    return {"U": Uo, "V": Vo, "W": Wo}
