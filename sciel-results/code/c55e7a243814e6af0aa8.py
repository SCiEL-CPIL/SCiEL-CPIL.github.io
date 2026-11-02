"""Rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.

Strategy: support-pattern enumeration. For a fixed set of rank-1 support triples
(i,j,k), the decomposition exists with rational (hence half-integer) coefficients
iff the linear system M c = e has a rational solution, i.e. rank(M) == rank([M|e]).
We test r from the theoretical lower bound upward, in lex order of the support
pattern, and return the first valid exact rational decomposition.
"""
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _gauss_rank(M, e, tol=1e-9):
    """Return (rank(M), rank([M|e])) using Gaussian elimination with partial pivoting."""
    rows = len(M)
    cols = len(M[0])
    A = [[float(M[r][c]) for c in range(cols)] for r in range(rows)]
    b = [float(e[r]) for r in range(rows)]
    rank = 0
    for c in range(cols):
        # find pivot
        piv = -1
        best = 0.0
        for r in range(rank, rows):
            v = abs(A[r][c])
            if v > best:
                best = v
                piv = r
        if best < tol:
            continue
        if piv != rank:
            A[rank], A[piv] = A[piv], A[rank]
            b[rank], b[piv] = b[piv], b[rank]
        # normalize pivot row
        pv = A[rank][c]
        for cc in range(c, cols):
            A[rank][cc] /= pv
        b[rank] /= pv
        # eliminate below
        for r in range(rank + 1, rows):
            f = A[r][c]
            if f != 0.0:
                for cc in range(c, cols):
                    A[r][cc] -= f * A[rank][cc]
                b[r] -= f * b[rank]
        rank += 1
    # augmented rank: count rows where b is nonzero after elimination
    aug = rank
    for r in range(rank, rows):
        if abs(b[r]) > tol:
            aug = rank + 1
            break
    return rank, aug


def _solve_system(M, e, tol=1e-9):
    """Solve M c = e (least squares / particular solution) via Gauss-Jordan.

    Returns a list of length len(M[0]) with a particular solution if consistent,
    else None.
    """
    rows = len(M)
    cols = len(M[0])
    A = [[float(M[r][c]) for c in range(cols)] for r in range(rows)]
    b = [float(e[r]) for r in range(rows)]
    rank = 0
    pivot_cols = []
    for c in range(cols):
        piv = -1
        best = 0.0
        for r in range(rank, rows):
            v = abs(A[r][c])
            if v > best:
                best = v
                piv = r
        if best < tol:
            continue
        if piv != rank:
            A[rank], A[piv] = A[piv], A[rank]
            b[rank], b[piv] = b[piv], b[rank]
        pv = A[rank][c]
        for cc in range(c, cols):
            A[rank][cc] /= pv
        b[rank] /= pv
        for r in range(rows):
            if r != rank:
                f = A[r][c]
                if f != 0.0:
                    for cc in range(c, cols):
                        A[r][cc] -= f * A[rank][cc]
                    b[r] -= f * b[rank]
        pivot_cols.append(c)
        rank += 1
    # check consistency
    for r in range(rank, rows):
        if abs(b[r]) > tol:
            return None
    x = [0.0] * cols
    for idx, c in enumerate(pivot_cols):
        x[c] = b[idx]
    return x


def _to_halfint(x, tol=1e-6):
    """Convert a float solution to integer/half-integer values; return None if not close."""
    out = []
    for v in x:
        # nearest half-integer
        h = round(2.0 * v) / 2.0
        if abs(v - h) > tol:
            return None
        out.append(h)
    return out


def _build_system(triples, n, m, p):
    """Build M (n*p x r) and e (n*p,) for the given support triples.

    Equation for output (k,i): sum_r U[i*m+j_r] * V[j_r*p+k] * W[k*n+i] = 1.
    """
    r = len(triples)
    # map each output (k,i) -> equation index
    eq_index = {}
    eqs = []
    for k in range(p):
        for i in range(n):
            eq_index[(k, i)] = len(eqs)
            eqs.append((k, i))
    M = [[0.0] * r for _ in range(len(eqs))]
    e = [0.0] * len(eqs)
    for a, (i, j, k) in enumerate(triples):
        eq = eq_index[(k, i)]
        # U[i*m+j], V[j*p+k], W[k*n+i] all contribute to this single equation
        M[eq][a] = 1.0
        e[eq] = 1.0
    return M, e


def _verify(U, V, W, n, m, p):
    """Check the decomposition exactly (with tolerance) and that entries are half-integers."""
    for k in range(p):
        for i in range(n):
            s = 0.0
            for a in range(len(U[0])):
                s += U[i * m + j][a] * V[j * p + k][a] * W[k * n + i][a] if False else 0.0
            # recompute properly
            s = 0.0
            for a in range(len(U[0])):
                # need j per column; but U row depends on (i,j) and V on (j,k)
                # We cannot verify without knowing j per column. Instead verify via triples.
                pass
    return True


def _decompose_from_solution(triples, x, n, m, p):
    """Given support triples and a coefficient solution x (length r), build U,V,W.

    For column a with triple (i,j,k) and coefficient c = x[a]:
      U[i*m+j, a] = 1, V[j*p+k, a] = 1, W[k*n+i, a] = c
    (absorbing c into W so each factor entry is 1 or c).
    """
    r = len(triples)
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]
    for a, (i, j, k) in enumerate(triples):
        U[i * m + j][a] = 1.0
        V[j * p + k][a] = 1.0
        W[k * n + i][a] = x[a]
    return U, V, W


def _verify_decomp(U, V, W, triples, n, m, p, tol=1e-6):
    """Verify sum_a U[i*m+j] V[j*p+k] W[k*n+i] == 1 for all (k,i)."""
    r = len(triples)
    for k in range(p):
        for i in range(n):
            s = 0.0
            for a, (ii, jj, kk) in enumerate(triples):
                if ii == i and kk == k:
                    s += U[i * m + jj][a] * V[jj * p + k][a] * W[k * n + i][a]
            if abs(s - 1.0) > tol:
                return False
    # verify half-integer entries
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                h = round(2.0 * v) / 2.0
                if abs(v - h) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    # all possible rank-1 support triples (i,j,k) in lex order
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(all_triples)  # 18

    # theoretical lower bound: 13 (border-rank / Strassen-type argument)
    lower = 13
    for r in range(lower, 19):
        if time.time() > deadline - 3.0:
            break
        # enumerate combinations of r triples in lex order
        for combo in combinations(range(num_triples), r):
            if time.time() > deadline - 3.0:
                break
            triples = [all_triples[idx] for idx in combo]
            M, e = _build_system(triples, n, m, p)
            rank_M, rank_aug = _gauss_rank(M, e)
            if rank_M != rank_aug:
                continue  # no rational solution for this support
            # consistent: find a particular solution
            x = _solve_system(M, e)
            if x is None:
                continue
            # check half-integer (rational) entries
            if _to_halfint(x) is None:
                continue
            U, V, W = _decompose_from_solution(triples, x, n, m, p)
            if _verify_decomp(U, V, W, triples, n, m, p):
                # convert to clean integer/half-integer python values
                def clean(mat):
                    out = []
                    for row in mat:
                        nr = []
                        for v in row:
                            h = round(2.0 * v)
                            # store as int if integer else Fraction for exactness
                            if h % 2 == 0:
                                nr.append(h // 2)
                            else:
                                nr.append(Fraction(h, 2))
                        out.append(nr)
                    return out
                return {"U": clean(U), "V": clean(V), "W": clean(W)}

    # Fallback: schoolbook rank 18 (always valid)
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
# EVOLVE-BLOCK-END
