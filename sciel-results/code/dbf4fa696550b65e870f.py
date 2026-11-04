import numpy as np
import time

DIMENSIONS = [3, 4, 6]


def _build_baseline():
    """Build the rank-66 decomposition.
    Returns U (n*m, r), V (m*p, r), W (p*n, r) as float arrays, plus r."""
    n, m, p = DIMENSIONS
    r0 = 66
    U = np.zeros((n * m, r0), dtype=float)
    V = np.zeros((m * p, r0), dtype=float)
    W = np.zeros((p * n, r0), dtype=float)
    col = 0

    def Aidx(i, j):
        return i * m + j

    def Bidx(j, k):
        return j * p + k

    def Cidx(i, k):
        return k * n + i

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c + 1)
        A10 = Aidx(a_r + 1, a_c); A11 = Aidx(a_r + 1, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c + 1)
        C10 = Cidx(c_r + 1, c_c); C11 = Cidx(c_r + 1, c_c + 1)
        # M1
        U[A00, col] += 1; U[A11, col] += 1; V[B00, col] += 1; V[B11, col] += 1; W[C00, col] += 1; W[C11, col] += 1; col += 1
        # M2
        U[A10, col] += 1; U[A11, col] += 1; V[B00, col] += 1; W[C10, col] += 1; W[C11, col] -= 1; col += 1
        # M3
        U[A00, col] += 1; V[B01, col] += 1; V[B11, col] -= 1; W[C01, col] += 1; W[C11, col] += 1; col += 1
        # M4
        U[A11, col] += 1; V[B10, col] += 1; V[B00, col] -= 1; W[C00, col] += 1; W[C10, col] += 1; col += 1
        # M5
        U[A00, col] += 1; U[A01, col] += 1; V[B11, col] += 1; W[C00, col] -= 1; W[C01, col] += 1; col += 1
        # M6
        U[A10, col] += 1; U[A00, col] -= 1; V[B00, col] += 1; V[B01, col] += 1; W[C11, col] += 1; col += 1
        # M7
        U[A01, col] += 1; U[A11, col] -= 1; V[B10, col] += 1; V[B11, col] += 1; W[C00, col] += 1; col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c + 1)
        U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1; col += 1
        U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1; col += 1
        U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1; col += 1
        U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1; col += 1

    # Row 0 (1x4 x 4x6): 6 blocks of 1x2 x 2x2
    add_1x2_2x2(0, 0, 0, 0, 0, 0); add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2); add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4); add_1x2_2x2(0, 2, 2, 4, 0, 4)
    # Rows 1-2 (2x4 x 4x6): 6 Strassen 2x2 x 2x2 blocks
    add_strassen(1, 0, 0, 0, 1, 0); add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2); add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4); add_strassen(1, 2, 2, 4, 1, 4)

    return U[:, :col], V[:, :col], W[:, :col], col


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _drop_zero_columns(U, V, W):
    """Drop columns that are identically zero in all three matrices."""
    r = U.shape[1]
    keep = []
    for l in range(r):
        if np.all(U[:, l] == 0) and np.all(V[:, l] == 0) and np.all(W[:, l] == 0):
            continue
        keep.append(l)
    if len(keep) == r:
        return U, V, W
    return U[:, keep], V[:, keep], W[:, keep]


def _try_fold(U, V, W, deadline):
    """Attempt to reduce rank by merging two rank-1 terms into one.
    A merge is valid if the 2-rank-1 sum of the two terms is itself rank-1
    in the (U,V,W) factorization sense: i.e., the 2x? matrix of coefficients
    for each of U,V,W is rank 1 (the two terms' supports are proportional).
    This is a conservative, exact check."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    best = (U, V, W)
    for a in range(r):
        for b in range(a + 1, r):
            if time.time() > deadline:
                break
            ua = U[:, a]; ub = U[:, b]
            va = V[:, a]; vb = V[:, b]
            wa = W[:, a]; wb = W[:, b]
            # Check proportionality: ua = alpha*ub, va = alpha*vb, wa = alpha*wb
            # Find alpha from a nonzero pair.
            def prop_ratio(x, y):
                # return alpha such that x = alpha*y, or None
                nz = np.nonzero(y)[0]
                if len(nz) == 0:
                    if np.all(x == 0):
                        return 0.0
                    return None
                ratios = x[nz] / y[nz]
                if not np.allclose(ratios, ratios[0], atol=1e-9):
                    return None
                # verify x is zero where y is zero
                if not np.allclose(x[np.nonzero(y == 0)[0]], 0, atol=1e-9):
                    return None
                return float(ratios[0])
            # We need a common scaling: actually the merge of two rank-1 terms
            # (u1 v1 w1) + (u2 v2 w2) is rank-1 only if (u1,u2),(v1,v2),(w1,w2)
            # are all proportional to the same vector. i.e. u2=c*u1, v2=c*v1, w2=c*w1.
            # Then sum = (1+c)*(u1 v1 w1) which is rank-1 with scalar (1+c).
            # So check u2 = c u1, v2 = c v1, w2 = c w1 for a single c.
            c = prop_ratio(ub, ua)
            if c is None:
                continue
            if prop_ratio(vb, va) != c:
                continue
            if prop_ratio(wb, wa) != c:
                continue
            # Merge: new term = (1+c) * (ua, va, wa)
            # Build new matrices without column b, with column a scaled by (1+c)
            keep = [l for l in range(r) if l != b]
            newU = U[:, keep].copy()
            newV = V[:, keep].copy()
            newW = W[:, keep].copy()
            ai = keep.index(a)
            newU[:, ai] *= (1.0 + c)
            newV[:, ai] *= (1.0 + c)
            newW[:, ai] *= (1.0 + c)
            if _verify(newU, newV, newW):
                return newU, newV, newW
    return best


def construct():
    start = time.time()
    deadline = start + 118.0  # safety margin before 150s kill

    U, V, W, r = _build_baseline()

    # Drop any zero columns (none expected, but safe)
    U, V, W = _drop_zero_columns(U, V, W)

    # Verify baseline
    if not _verify(U, V, W):
        # Should not happen; return baseline anyway (best effort)
        return {"U": U, "V": V, "W": W}

    bestU, bestV, bestW = U, V, W
    best_rank = U.shape[1]

    # Local search: repeatedly try to fold (merge) pairs of rank-1 terms.
    # Each successful fold reduces rank by 1.
    while time.time() < deadline:
        newU, newV, newW = _try_fold(U, V, W, deadline)
        if newU.shape[1] < U.shape[1]:
            U, V, W = newU, newV, newW
            U, V, W = _drop_zero_columns(U, V, W)
            if U.shape[1] < best_rank:
                bestU, bestV, bestW = U, V, W
                best_rank = U.shape[1]
        else:
            break  # no fold found, stop

    # Final safety: ensure we return a valid decomposition
    if not _verify(bestU, bestV, bestW):
        bestU, bestV, bestW = U, V, W
        if not _verify(bestU, bestV, bestW):
            # fallback to fresh baseline
            bestU, bestV, bestW, _ = _build_baseline()

    return {"U": bestU, "V": bestV, "W": bestW}
