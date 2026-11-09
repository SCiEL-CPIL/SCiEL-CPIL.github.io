"""Search for a rank-minimal exact integer tensor decomposition of 2x4 * 4x7."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
_n, _m, _p = DIMENSIONS
_N, _M, _P = _n * _m, _m * _p, _p * _n  # 8, 28, 14
_SCHOOL = _n * _m * _p  # 56

def _build_tensor():
    # T[i*m+j, j*p+k, k*n+i] = 1  (index: U-row, V-row, W-row)
    T = {}
    for i in range(_n):
        for j in range(_m):
            for k in range(_p):
                T[(i * _m + j, j * _p + k, k * _n + i)] = Fraction(1)
    return T

_T = _build_tensor()
_KEYS = list(_T.keys())

def _schoolbook():
    rank = _SCHOOL
    U = [[0] * rank for _ in range(_N)]
    V = [[0] * rank for _ in range(_M)]
    W = [[0] * rank for _ in range(_P)]
    c = 0
    for i in range(_n):
        for j in range(_m):
            for k in range(_p):
                U[i * _m + j][c] = 1
                V[j * _p + k][c] = 1
                W[k * _n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _decompose_from_factors(U, V, W):
    """U: list[N][r], V: list[M][r], W: list[P][r] (Fraction). Return dict or None."""
    r = len(U[0])
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    r = len(U[0])
    for (a, b, c), val in _T.items():
        s = Fraction(0)
        for t in range(r):
            s += U[a][t] * V[b][t] * W[c][t]
        if s != val:
            return False
    return True

def _search_rank(r, deadline, rng):
    """Try to find integer/half-integer U,V,W of rank r. Return dict or None."""
    # Strategy: random W with small half-integers, then solve for U,V via
    # a bilinear alternating least squares over Fractions.
    # We build the linear system: for each (a,b,c) in support, sum_t U[a,t] V[b,t] W[c,t] = 1.
    # Fix W randomly, then the system is bilinear in (U,V). Use ALS:
    #   given V, solve for U (linear, since terms are U[a,t]* (V[b,t]W[c,t])).
    #   given U, solve for V.
    # This is a rank-r factorization of a 3-tensor via alternating least squares
    # on the bilinear form. We solve the linear systems exactly with Fractions.

    # Build coefficient maps.
    # For U update: unknowns U[a,t] for a in [0,N), t in [0,r).
    # Equation for (a,b,c): sum_t U[a,t] * (V[b,t]*W[c,t]) = 1.
    # This is linear in U[a,*] only! Because each equation involves only U[a,*]
    # (a fixed). So U decomposes per-row: for each a, we have equations over (b,c).
    # Similarly V per-row, W per-row.
    # So ALS per-row linear solves are independent and exact.

    # Initialize W randomly with half-integers in {-1,-1/2,0,1/2,1}.
    W = [[Fraction(rng.choice([-2,-1,0,1,2])) / 2 for _ in range(r)] for _ in range(_P)]

    # Initialize U,V randomly.
    U = [[Fraction(rng.choice([-2,-1,0,1,2])) / 2 for _ in range(r)] for _ in range(_N)]
    V = [[Fraction(rng.choice([-2,-1,0,1,2])) / 2 for _ in range(r)] for _ in range(_M)]

    # Precompute which (b,c) pairs appear for each a, and (a,c) for each b, (a,b) for each c.
    by_a = {}
    by_b = {}
    by_c = {}
    for (a, b, c) in _KEYS:
        by_a.setdefault(a, []).append((b, c, _T[(a, b, c)]))
        by_b.setdefault(b, []).append((a, c, _T[(a, b, c)]))
        by_c.setdefault(c, []).append((a, b, _T[(a, b, c)]))

    def _solve_linear(rows, coeffs, rhs, var_count):
        """Solve A x = rhs where A is rows x var_count (Fraction), via Gaussian elim."""
        # Build augmented matrix
        M = [[coeffs[i][j] for j in range(var_count)] + [rhs[i]] for i in range(len(rows))]
        # Gaussian elimination
        n = var_count
        m = len(rows)
        rank = 0
        col_used = [False] * n
        for col in range(n):
            # find pivot
            pivot = None
            for row in range(rank, m):
                if M[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            M[rank], M[pivot] = M[pivot], M[rank]
            pv = M[rank][col]
            # normalize
            for j in range(col, n + 1):
                M[rank][j] = M[rank][j] / pv
            for row in range(m):
                if row != rank and M[row][col] != 0:
                    factor = M[row][col]
                    for j in range(col, n + 1):
                        M[row][j] -= factor * M[rank][j]
            rank += 1
        # Extract solution: if consistent, x[col] from row
        x = [Fraction(0)] * n
        for row in range(rank):
            # find leading col
            lc = None
            for col in range(n):
                if M[row][col] != 0:
                    lc = col
                    break
            if lc is not None:
                x[lc] = M[row][n]
        # Check consistency
        for row in range(rank, m):
            if M[row][n] != 0:
                return None  # inconsistent
        return x

    for it in range(60):
        if time.time() > deadline:
            return None
        # Update U: for each a, solve for U[a,*] given V,W.
        for a in range(_N):
            eqs = by_a[a]
            # unknowns U[a,0..r-1]; coefficient of U[a,t] is V[b,t]*W[c,t]
            rows = []
            coeffs = []
            rhs = []
            for (b, c, val) in eqs:
                rows.append(1)
                coeffs.append([V[b][t] * W[c][t] for t in range(r)])
                rhs.append(val)
            sol = _solve_linear(rows, coeffs, rhs, r)
            if sol is None:
                return None
            U[a] = sol
        # Update V
        for b in range(_M):
            eqs = by_b[b]
            coeffs = []
            rhs = []
            for (a, c, val) in eqs:
                coeffs.append([U[a][t] * W[c][t] for t in range(r)])
                rhs.append(val)
            sol = _solve_linear([1] * len(eqs), coeffs, rhs, r)
            if sol is None:
                return None
            V[b] = sol
        # Update W
        for c in range(_P):
            eqs = by_c[c]
            coeffs = []
            rhs = []
            for (a, b, val) in eqs:
                coeffs.append([U[a][t] * V[b][t] for t in range(r)])
                rhs.append(val)
            sol = _solve_linear([1] * len(eqs), coeffs, rhs, r)
            if sol is None:
                return None
            W[c] = sol
        # Check exactness
        if _verify(U, V, W):
            # convert to float-free representation; keep as Fraction numerators/2
            Uf = [[(x * 2).numerator / (x * 2).denominator if (x * 2).denominator != 1 else float((x * 2).numerator) for x in row] for row in U]
            Vf = [[(x * 2).numerator / (x * 2).denominator if (x * 2).denominator != 1 else float((x * 2).numerator) for x in row] for row in V]
            Wf = [[(x * 2).numerator / (x * 2).denominator if (x * 2).denominator != 1 else float((x * 2).numerator) for x in row] for row in W]
            return {"U": Uf, "V": Vf, "W": Wf}
    return None

def construct():
    start = time.time()
    deadline = start + 100  # safety margin before 120s
    school = _schoolbook()
    best = school
    best_rank = _SCHOOL
    rng = random.Random(12345)
    # Try ranks from a lower bound up to schoolbook-1.
    # Lower bound: max(n*p, m*p, n*m) = max(14,28,8)=28
    lower = max(_n * _p, _m * _p, _n * _m)
    for r in range(lower, _SCHOOL):
        if time.time() > deadline:
            break
        # a few restarts per rank
        found = None
        for attempt in range(40):
            if time.time() > deadline:
                break
            found = _search_rank(r, deadline, rng)
            if found is not None:
                break
        if found is not None:
            best = found
            best_rank = r
            break  # found minimal rank in this ascending scan
    return best

# EVOLVE-BLOCK-END
