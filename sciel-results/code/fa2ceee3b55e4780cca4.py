"""Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """3-mode tensor T[i,j,k] = 1 iff (i,j,k) is a valid multiplication term."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = Fraction(1)
    return T


def _contract(T, U, V, W):
    """Exact contraction C[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]."""
    C = {}
    for (i, j, k), val in T.items():
        s = Fraction(0)
        for c in range(len(U)):
            s += U[i, j, c] * V[j, k, c] * W[k, i, c]
        C[(i, j, k)] = s
    return C


def _verify(T, U, V, W):
    """Return True iff the decomposition exactly reconstructs T."""
    C = _contract(T, U, V, W)
    for key, val in T.items():
        if C.get(key, Fraction(0)) != val:
            return False
    return True


def _to_int_matrix(U, V, W):
    """Convert Fraction matrices to integer (half-integer safe) lists."""
    def conv(M):
        rows = len(M)
        cols = len(M[0]) if rows else 0
        out = [[0] * cols for _ in range(rows)]
        for a in range(rows):
            for b in range(cols):
                v = M[a][b]
                # entries are integers or half-integers; keep as int if integral
                if v.denominator == 1:
                    out[a][b] = int(v)
                else:
                    out[a][b] = v  # Fraction (half-integer) is acceptable
        return out
    return conv(U), conv(V), conv(W)


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    T = _build_tensor(n, m, p)
    I = list(range(n))
    J = list(range(m))
    K = list(range(p))

    # Initialize with the schoolbook rank-1 components (rank = n*m*p).
    U = {}  # (i,j) -> list of Fractions
    V = {}  # (j,k) -> list of Fractions
    W = {}  # (k,i) -> list of Fractions
    for i in I:
        for j in J:
            U[(i, j)] = [Fraction(0)]
    for j in J:
        for k in K:
            V[(j, k)] = [Fraction(0)]
    for k in K:
        for i in I:
            W[(k, i)] = [Fraction(0)]
    for i in I:
        for j in J:
            for k in K:
                U[(i, j)][0] = Fraction(1)
                V[(j, k)][0] = Fraction(1)
                W[(k, i)][0] = Fraction(1)

    def to_dense(M, keys):
        """list of columns (rank) as list of dict key->Fraction."""
        rank = len(next(iter(M.values())))
        cols = [dict() for _ in range(rank)]
        for key, col in M.items():
            for c in range(rank):
                if col[c] != 0:
                    cols[c][key] = col[c]
        return cols

    def norm2(M, keys):
        s = Fraction(0)
        for key in keys:
            v = M[key]
            for c in v:
                s += v[c] * v[c]
        return s

    best_U, best_V, best_W = U, V, W

    # Greedy rank reduction: try to eliminate a rank-1 component each step.
    # A component c is removable if for every (i,j,k) the product U*V*W has
    # zero contribution from c OR the residual after removing c still matches.
    # We attempt to zero out a column by finding a (i,j,k) where only that
    # column is nonzero in the product, then scale to cancel.
    def current_rank():
        return len(next(iter(U.values())))

    def try_eliminate(cols_U, cols_V, cols_W, c):
        """Attempt to remove column c by exact cancellation. Returns (U,V,W) new
        matrices with column c removed, or None if not possible."""
        # For column c to be removable, there must exist a pivot (i0,j0,k0)
        # where the product P_c = U[i0,j0,c]*V[j0,k0,c]*W[k0,i0,c] != 0 and
        # for all other columns d != c, P_d == 0 at this pivot, so we can
        # absorb column c into others... Actually simpler: we just drop column c
        # if it contributes zero to the contraction everywhere.
        # Check: does column c contribute anywhere?
        contrib = False
        for i in I:
            for j in J:
                for k in K:
                    prod = U[(i, j)][c] * V[(j, k)][c] * W[(k, i)][c]
                    if prod != 0:
                        contrib = True
        if not contrib:
            # safe to drop
            def drop(M, c):
                out = {}
                for key, col in M.items():
                    newcol = [x for idx, x in enumerate(col) if idx != c]
                    if newcol:
                        out[key] = newcol
                return out
            return drop(U, c), drop(V, c), drop(W, c)
        return None

    # Main loop: repeatedly try to find a removable column.
    rank = current_rank()
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for c in range(rank):
            res = try_eliminate(None, None, None, c)
            if res is not None:
                U, V, W = res
                rank = current_rank()
                progress = True
                break

    # Final exact verification
    if not _verify(T, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = {(i, j): [Fraction(0)] * rank for i in I for j in J}
        V = {(j, k): [Fraction(0)] * rank for j in J for k in K}
        W = {(k, i): [Fraction(0)] * rank for k in K for i in I}
        col = 0
        for i in I:
            for j in J:
                for k in K:
                    U[(i, j)][col] = Fraction(1)
                    V[(j, k)][col] = Fraction(1)
                    W[(k, i)][col] = Fraction(1)
                    col += 1

    rank = current_rank()
    U_list = [[int(U[(i, j)][c]) if U[(i, j)][c].denominator == 1 else U[(i, j)][c]
               for c in range(rank)] for i in I for j in J]
    V_list = [[int(V[(j, k)][c]) if V[(j, k)][c].denominator == 1 else V[(j, k)][c]
               for c in range(rank)] for j in J for k in K]
    W_list = [[int(W[(k, i)][c]) if W[(k, i)][c].denominator == 1 else W[(k, i)][c]
               for c in range(rank)] for k in K for i in I]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
