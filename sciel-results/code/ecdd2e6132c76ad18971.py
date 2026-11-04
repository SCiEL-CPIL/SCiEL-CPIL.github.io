"""Rank-minimal exact tensor decomposition for 4x5 by 5x6 matrix multiplication."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [4, 5, 6]
N, M, P = DIMENSIONS
SCHOOLBOOK_RANK = N * M * P  # 120


def _schoolbook():
    n, m, p = N, M, P
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _validate(U, V, W):
    n, m, p = N, M, P
    r = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(r):
                        u = U[i * m + j][c]
                        w = W[k * n + i][c]
                        v = V[jp * p + k][c]
                        if u and w and v:
                            s += u * w * v
                    if s != (1 if j == jp else 0):
                        return False
    return True


def _solve_linear(A, b):
    """Solve A x = b exactly, A is (n x y) of ints, b is (n,) of ints.
    Returns x as list of Fraction of length y, or None if no/infinite unique issue.
    Uses fraction-free-ish Gauss-Jordan with Fraction."""
    n = len(A)
    y = len(A[0]) if n else 0
    aug = [[Fraction(A[i][j]) for j in range(y)] + [Fraction(b[i])] for i in range(n)]
    row = 0
    pivot_col = []
    for col in range(y):
        if row >= n:
            break
        piv = -1
        best = None
        for i in range(row, n):
            v = aug[i][col]
            if v == 0:
                continue
            mag = abs(v)
            if best is None or mag > best:
                best = mag
                piv = i
        if piv == -1:
            continue
        if piv != row:
            aug[row], aug[piv] = aug[piv], aug[row]
        pv = aug[row][col]
        inv = Fraction(1) / pv
        aug[row] = [v * inv for v in aug[row]]
        for i in range(n):
            if i == row:
                continue
            f = aug[i][col]
            if f == 0:
                continue
            aug[i] = [aug[i][j] - f * aug[row][j] for j in range(y + 1)]
        pivot_col.append(col)
        row += 1
    # Check consistency: any all-zero-row-in-A must have b=0
    for i in range(n):
        allzero = True
        for j in range(y):
            if aug[i][j] != 0:
                allzero = False
                break
        if allzero and aug[i][y] != 0:
            return None
    x = [Fraction(0)] * y
    for i in range(row):
        # find pivot col of this row
        for j in range(y):
            if aug[i][j] != 0:
                x[j] = aug[i][y]
                break
    return x


def _try_rank(r, seed):
    n, m, p = N, M, P
    rng = random.Random(seed)
    U = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
    V = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for i in range(n):
        for k in range(p):
            Aeq = []
            beq = []
            for j in range(m):
                for jp in range(m):
                    row = [U[i * m + j][c] * V[jp * p + k][c] for c in range(r)]
                    Aeq.append(row)
                    beq.append(1 if j == jp else 0)
            x = _solve_linear(Aeq, beq)
            if x is None:
                return None
            W[k * n + i] = [x[c].numerator if x[c].denominator == 1 else x[c] for c in range(r)]
    # Validate
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0
                    for c in range(r):
                        u = U[i * m + j][c]
                        w = W[k * n + i][c]
                        v = V[jp * p + k][c]
                        if u and w and v:
                            s += u * w * v
                    if s != (1 if j == jp else 0):
                        return None
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 110.0
    best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK

    # Try a few ranks below schoolbook, limited seeds, with time guard.
    # Focus effort on small-to-mid ranks where a solution is plausible-ish.
    rank_order = []
    # Try ranks from a modest set, descending, but cap total attempts.
    for r in range(min(SCHOOLBOOK_RANK - 1, 60), 0, -1):
        rank_order.append(r)
    # Limit to first K ranks to respect budget
    K = 25
    for r in rank_order[:K]:
        if time.time() > deadline:
            break
        for s in range(2):
            if time.time() > deadline:
                break
            res = _try_rank(r, seed=5000 + r * 13 + s)
            if res is not None:
                best = res
                best_rank = r
                return best
    return best


if __name__ == "__main__":
    res = construct()
    r = len(res["U"][0])
    print("rank:", r)
    ok = _validate(res["U"], res["V"], res["W"])
    print("valid:", ok)
    if not ok:
        sys.exit(1)
