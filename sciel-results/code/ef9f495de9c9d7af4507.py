"""SCiEL task: rank-minimal tensor decomposition for 4x4 * 4x9 matmul.

Approach family (distinct from seed): a runtime bilinear-algebra solver.
  * Enumerate candidate ranks r from a provable lower bound up to schoolbook.
  * For each r, solve the homogeneous linear system  U⊗V·vec(T) = 0  for the
    third factor W (trace convention W[k*n+i, c]).
  * Randomised multi-start + Gaussian elimination (exact rational via fractions).
  * Deterministically verify every emitted factor against the matmul tensor.
  * Fallback to the guaranteed schoolbook (rank n*m*p) if the search fails.
"""
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _mul3(n, m, p):
    # 3-way tensor t[i,m + j,p + k] = e_{i,k}  (a[i,j] b[j,k] -> C[i,k])
    t = {}
    for i in range(n):
        for k in range(p):
            for j in range(m):
                t[i * m + j, j * p + k, k * n + i] = 1
    return t


def _tvec(t, n, m, p):
    # vec over (i,j) rows, (k) cols
    b = n * m * p
    return [t.get((i, j, k), 0) for i in range(n) for j in range(m) for k in range(p)]


def _gauss(A, b):
    # Solve A x = b over rationals; return (x or None, pivot_cols)
    n = len(b)
    m = len(A[0])
    M = [[Fraction(x) for x in row] + [Fraction(b[i])] for i, row in enumerate(A)]
    r = 0
    pivots = []
    for col in range(m):
        piv = -1
        for i in range(r, n):
            if M[i][col] != 0:
                piv = i
                break
        if piv == -1:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][col]
        for c in range(col, m + 1):
            M[r][c] /= pv
        for i in range(n):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                for c in range(col, m + 1):
                    M[i][c] -= f * M[r][c]
        pivots.append(col)
        r += 1
    for i in range(r, n):
        if M[i][m] != 0:
            return None, pivots
    x = [Fraction(0)] * m
    for i, pc in enumerate(pivots):
        x[pc] = M[i][m]
    return x, pivots


def _hom(A):
    # Basis of null space of A (rows x cols) over rationals.
    n, m = len(A), len(A[0])
    M = [[Fraction(x) for x in row] for row in A]
    r = 0
    pivots = []
    for col in range(m):
        piv = -1
        for i in range(r, n):
            if M[i][col] != 0:
                piv = i
                break
        if piv == -1:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][col]
        for c in range(col, m):
            M[r][c] /= pv
        for i in range(n):
            if i != r and M[i][col] != 0:
                f = M[i][col]
                for c in range(col, m):
                    M[i][c] -= f * M[r][c]
        pivots.append(col)
        r += 1
    free = [c for c in range(m) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * m
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -M[i][fc]
        basis.append(v)
    return basis, pivots


def _scale(v):
    nz = [x for x in v if x != 0]
    if not nz:
        return None
    def g(a, b):
        while b:
            a, b = b, a % b
        return a
    l = abs(nz[0].denominator)
    for x in nz[1:]:
        l = l * x.denominator // g(l, x.denominator)
    return [x * l for x in v]


def _buildW(B, t, n, m, p, r):
    # Unknowns in W-major order: for each r, for each k, for each i -> (k*n+i)
    Wn = p * n
    b = n * m * p
    rows = []
    rhs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = [0] * (Wn * r)
                for c in range(r):
                    idx = i * m + j
                    val = B[idx][c]
                    if val != 0:
                        row[(k * n + i) * r + c] = val
                rows.append(row)
                rhs.append(t.get((i, j, k), 0))
    x, _ = _gauss(rows, rhs)
    if x is None:
        return None
    W = [0] * (Wn * r)
    for c in range(r):
        for i in range(n):
            for k in range(p):
                W[(k * n + i) * r + c] = x[(k * n + i) * r + c]
    return W


def _solve(r, t, n, m, p, seed):
    rng = random.Random(seed)
    A = n * m
    B = m * p
    for trial in range(60):
        U = [[Fraction(rng.randint(-2, 2)) for _ in range(r)] for _ in range(A)]
        Bf = [[Fraction(rng.randint(-2, 2)) for _ in range(r)] for _ in range(B)]
        W = _buildW(Bf, t, n, m, p, r)
        if W is None:
            continue
        # verify U⊗V·vec = vec(T)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    ui = i * m + j
                    bj = j * p + k
                    for c in range(r):
                        s += U[ui][c] * Bf[bj][c] * W[(k * n + i) * r + c]
                    if s != Fraction(t.get((i, j, k), 0)):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return U, Bf, W
    return None


def construct():
    n, m, p = DIMENSIONS
    t = _mul3(n, m, p)
    tvec = _tvec(t, n, m, p)

    # Lower bound: rank >= max(n*p, m*p) ; refine via generic rank of slices.
    A = n * m
    B = m * p
    Wn = p * n
    M = [[tvec[i * p + k] for k in range(p)] for i in range(A)]
    lb = 0
    for cand in range(1, A + 1):
        sub = [row[:cand] for row in M]
        basis, _ = _hom(sub)
        if len(basis) == 0:
            lb = cand
            break
    lb = max(lb, n * p, m * p)

    school = n * m * p
    for r in range(lb, school + 1):
        for sd in range(0, 60):
            sol = _solve(r, t, n, m, p, 1000003 + r * 97 + sd)
            if sol is None:
                continue
            U, V, W = sol
            # exact re-verification (guard against any arithmetic slip)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[(k * n + i) * r + c]
                        if s != Fraction(t.get((i, j, k), 0)):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                # emit as integer / half-integer lists
                def norm(Mat, Wrow):
                    out = []
                    for row in Mat:
                        nr = []
                        for x in row:
                            x = Fraction(x)
                            d = x.denominator
                            num = x.numerator * d  # = 2*(numerator/2) when d==2
                            nr.append(num / 2.0)
                        out.append(nr)
                    return out
                return {
                    "U": norm(U, None),
                    "V": norm(V, None),
                    "W": norm([[W[(k * n + i) * r + c] for c in range(r)]
                               for k in range(p) for i in range(n)], None),
                }

    # Guaranteed fallback: schoolbook (rank n*m*p).
    rank = school
    U = [[0.0] * rank for _ in range(A)]
    V = [[0.0] * rank for _ in range(B)]
    W = [[0.0] * rank for _ in range(Wn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
