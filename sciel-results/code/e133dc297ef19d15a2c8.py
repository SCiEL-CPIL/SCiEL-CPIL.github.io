"""Tensor rank search for 2x3x9 matrix multiplication.

Strategy:
- The multiplication tensor T has entries T[(i,j),(j,k),(k,i)] = 1 (with the
  trace convention W indexed by (k*n+i, r)).
- A rank-r decomposition is T = sum_{c} u_c (x) v_c (y) w_c (z), where
    U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r),
    and T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
- We search for the smallest r for which an exact (integer/half-integer)
    decomposition exists, using a randomized/structured ansatz plus an exact
    linear-algebra solve for the third factor given the first two, then
    verification with exact rational arithmetic (fractions) to avoid FP error.
"""

import time
import random
import itertools
from fractions import Fraction
from collections import defaultdict

import numpy as np

DIMENSIONS = [2, 3, 9]


def _build_tensor(n, m, p):
    """Return dict mapping (a,b,c) -> 1 for the multiplication tensor.
    a = i*m + j, b = j*p + k, c = k*n + i.
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T


def _verify_exact(U, V, W, T, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    # Build W as a 2D array of Fractions
    Wf = [[Fraction(W[r][c]) for c in range(p * n)] for r in range(rank)]
    Uf = [[Fraction(U[r][c]) for c in range(n * m)] for r in range(rank)]
    Vf = [[Fraction(V[r][c]) for c in range(m * p)] for r in range(rank)]
    for (a, b, c), val in T.items():
        s = Fraction(0)
        for r in range(rank):
            s += Uf[r][a] * Vf[r][b] * Wf[r][c]
        if s != Fraction(val):
            return False
    return True


def _try_rank(n, m, p, r, T, rng, max_trials=200, deadline=None):
    """Try to find a rank-r decomposition.

    Approach: fix U and V with a structured/randomized ansatz over a small
    finite field of half-integers, then solve for W exactly via linear
    equations, and check that W entries are half-integers and that the
    decomposition reproduces T.
    """
    nm = n * m
    mp = m * p
    pn = p * n

    # Each trial: random U, V entries from a small set, solve for W.
    # The equation: for all (a,b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    # For a fixed (a,b), this is a linear system in the vector W[:,c] over c,
    # but W entries are shared across c. Instead, treat W as unknown matrix
    # (r x pn). For each (a,b) pair, we get pn linear equations in the r*pn
    # unknowns? That's coupled. Better: for each c, we have nm*mp equations:
    #   sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]  for all (a,b)
    # This is a linear system in the r unknowns W[c,0..r-1] for fixed c:
    #   (U^T * V) ... actually define M_{(a,b), r} = U[a,r]*V[b,r], then
    #   M_c * W[:,c] = T[:,c] where (M_c)_{(a,b),r} = U[a,r]*V[b,r].
    # M is the same for all c (since it doesn't depend on c)! So M is a
    # (nm*mp) x r matrix. We need M * w_c = t_c for each c, where w_c is the
    # r-vector of W[:,c] and t_c is the (nm*mp)-vector of T[:,c].
    # This is solvable if M has rank r and t_c is in the column space.
    # We solve via least squares over rationals, but M is fixed per trial.
    # Actually M depends on U and V but not c. Good.

    # Build M as rational matrix (nm*mp) x r
    # We'll use Fraction for exactness.
    # Choose U, V entries from a small set of half-integers.
    entry_set = [Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)]

    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            return None
        # Random U, V
        U = [[rng.choice(entry_set) for _ in range(r)] for _ in range(nm)]
        V = [[rng.choice(entry_set) for _ in range(r)] for _ in range(mp)]

        # Build M: rows indexed by (a,b), cols by r. M[(a,b), r] = U[a,r]*V[b,r]
        # We need M to have full column rank r. Check via exact rank (Gaussian elim).
        # Represent M as list of rows (each a list of r Fractions), length nm*mp.
        M = []
        for a in range(nm):
            for b in range(mp):
                row = [U[a][rr] * V[b][rr] for rr in range(r)]
                M.append(row)

        # Exact Gaussian elimination to compute rank and solve.
        # We'll do fraction-based elimination on the augmented system for each c.
        # First, check rank of M == r.
        # Perform elimination on M (as a matrix) to get row-echelon and track pivot cols.
        # To save time, we can compute rank of M once per trial.
        # Build a copy for elimination.
        A = [row[:] for row in M]
        ncols = r
        nrows = len(A)
        pivot_cols = []
        r_idx = 0
        for c in range(ncols):
            # find pivot
            piv = None
            for i in range(r_idx, nrows):
                if A[i][c] != 0:
                    piv = i
                    break
            if piv is None:
                continue
            A[r_idx], A[piv] = A[piv], A[r_idx]
            # normalize pivot row
            pv = A[r_idx][c]
            A[r_idx] = [x / pv for x in A[r_idx]]
            # eliminate
            for i in range(nrows):
                if i != r_idx and A[i][c] != 0:
                    factor = A[i][c]
                    A[i] = [x - factor * A[r_idx][x] for x in range(ncols)]
            pivot_cols.append(c)
            r_idx += 1
            if r_idx == ncols:
                break
        if r_idx != r:
            # M does not have full column rank; cannot uniquely solve. Skip.
            continue

        # Now solve M w_c = t_c for each c.
        # Since M has full column rank, we can use the reduced row-echelon form.
        # But we eliminated M alone; we need to also eliminate the RHS.
        # Let's redo elimination on augmented [M | t_c] for each c, but that's
        # pn times. Alternatively, compute a right-inverse-like solution.
        # Simpler: for each c, do elimination on augmented matrix.
        # pn = p*n = 18, r is small. nm*mp = 6*27=162 rows. 18 * 162*r ops.
        # For r up to ~30, that's fine.
        W = [[Fraction(0)] * pn for _ in range(r)]
        ok = True
        for c in range(pn):
            # Build t_c: for each (a,b), T[a,b,c]
            # Augmented: [M | t_c], solve for w (r unknowns).
            Aug = []
            for idx, (a, b) in enumerate(itertools.product(range(nm), range(mp))):
                # T value
                # a = i*m+j, b = j'*p+k. But T is only 1 when a,b,c consistent.
                # Use T dict.
                val = T.get((a, b, c), 0)
                row = [M[idx][rr] for rr in range(r)] + [Fraction(val)]
                Aug.append(row)
            # Gaussian elimination on Aug (rows x (r+1))
            A2 = [row[:] for row in Aug]
            nrow = len(A2)
            ncol = r + 1
            pivots = []
            rr = 0
            for ccol in range(r):
                piv = None
                for i in range(rr, nrow):
                    if A2[i][ccol] != 0:
                        piv = i
                        break
                if piv is None:
                    continue
                A2[rr], A2[piv] = A2[piv], A2[rr]
                pv = A2[rr][ccol]
                A2[rr] = [x / pv for x in A2[rr]]
                for i in range(nrow):
                    if i != rr and A2[i][ccol] != 0:
                        f = A2[i][ccol]
                        A2[i] = [x - f * A2[rr][x] for x in range(ncol)]
                pivots.append((rr, ccol))
                rr += 1
            if rr != r:
                ok = False
                break
            # Check consistency: any row with all-zero LHS must have zero RHS.
            for i in range(rr, nrow):
                if any(A2[i][x] != 0 for x in range(r)):
                    ok = False
                    break
                if A2[i][r] != 0:
                    ok = False
                    break
            if not ok:
                break
            # Solution: w[ccol] = A2[row][r] for each pivot (row,ccol)
            w = [Fraction(0)] * r
            for (row, ccol) in pivots:
                w[ccol] = A2[row][r]
            for rr2 in range(r):
                W[rr2][c] = w[rr2]
        if not ok:
            continue

        # Check W entries are half-integers (denominator in {1,2}).
        # Our entry_set for U,V are half-integers; W came from exact solve.
        # We need W entries to be half-integers too (task allows integer/half-integer).
        # Check each W entry: denominator must be 1 or 2.
        halfint_ok = True
        for rr2 in range(r):
            for c in range(pn):
                den = W[rr2][c].denominator
                if den not in (1, 2):
                    halfint_ok = False
                    break
            if not halfint_ok:
                break
        if not halfint_ok:
            continue

        # Convert to plain floats/ints for return, but verify exact first.
        Ulist = [[float(x) for x in row] for row in U]
        Vlist = [[float(x) for x in row] for row in V]
        Wlist = [[float(x) for x in row] for row in W]

        # Exact verify
        if _verify_exact(Ulist, Vlist, Wlist, T, n, m, p):
            return {"U": Ulist, "V": Vlist, "W": Wlist}
    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    deadline = time.time() + 120.0

    # Lower bound: rank >= max(n*m, m*p, p*n) / something. Known: rank >= max(n,m,p)
    # For 2x3x9, known rank is 27 (border rank). Try from a reasonable lower bound.
    # Use a safe lower bound: max(n*m, m*p, p*n) is 54 (schoolbook). We know rank < 54.
    # Start from a heuristic lower bound and go up.
    # Lower bound for matmul rank: rank >= max(n*m, m*p, p*n) is NOT true.
    # Actual lower bound: rank >= max(n, m, p) * something. Use rank >= max(n*m, m*p, p*n)/max(n,m,p) ...
    # Simple: try r from 20 upward to 54.
    rng = random.Random(12345)

    best = None
    for r in range(20, 55):
        if time.time() > deadline - 5.0:
            break
        res = _try_rank(n, m, p, r, T, rng, max_trials=300, deadline=deadline)
        if res is not None:
            best = (r, res)
            break  # smallest r found; return it
    if best is not None:
        return best[1]

    # Fallback: schoolbook rank 54
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
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U, V, W = res["U"], res["V"], res["W"]
    rank = len(U[0])
    print("rank =", rank)
    T = _build_tensor(n, m, p)
    ok = _verify_exact(U, V, W, T, n, m, p)
    print("valid =", ok)
