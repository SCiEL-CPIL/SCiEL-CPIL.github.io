"""Combined: deterministic rank-66 Strassen baseline + strong rank-reduction machinery.
Strategy: build the rank-66 baseline, then aggressively reduce rank by (a) dropping
zero-contribution columns, (b) merging proportional columns, and (c) eliminating
columns that are linear combinations of the others (via least-squares on the tensor
equations). Iterate until no further reduction is possible, then verify exactness.
"""
import time
import numpy as np

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p


def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # trace convention W[k*n+i]


def baseline_66():
    """Deterministic rank-66 decomposition (Strassen blocks + 1x2/2x2 blocks)."""
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c + 1)
        A10 = Aidx(a_r + 1, a_c); A11 = Aidx(a_r + 1, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c + 1)
        C10 = Cidx(c_r + 1, c_c); C11 = Cidx(c_r + 1, c_c + 1)
        # M1
        U[A00][col] += 1; U[A11][col] += 1; V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1; col += 1
        # M2
        U[A10][col] += 1; U[A11][col] += 1; V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1; col += 1
        # M3
        U[A00][col] += 1; V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1; col += 1
        # M4
        U[A11][col] += 1; V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1; col += 1
        # M5
        U[A00][col] += 1; U[A01][col] += 1; V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1; col += 1
        # M6
        U[A10][col] += 1; U[A00][col] -= 1; V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1; col += 1
        # M7
        U[A01][col] += 1; U[A11][col] -= 1; V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c + 1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c + 1)
        B10 = Bidx(b_r + 1, b_c); B11 = Bidx(b_r + 1, b_c + 1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c + 1)
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1; col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1; col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1; col += 1

    add_1x2_2x2(0, 0, 0, 0, 0, 0); add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2); add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4); add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0); add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2); add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4); add_strassen(1, 2, 2, 4, 1, 4)
    assert col == 66
    return U, V, W


def tensor_from(U, V, W):
    """Compute the full (n,m,p) tensor from factors U,V,W."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    return np.einsum('ijl,jkl,kil->ijk', U3, V3, W3)


def verify(U, V, W, tol=1e-9):
    """Verify exact decomposition: T[i,j,k] == 1 for all i,j,k."""
    T = tensor_from(U, V, W)
    return bool(np.allclose(T, 1.0, atol=tol))


def try_drop_zero(U, V, W):
    """Return index of a column whose contribution is identically zero, or None."""
    r = U.shape[1]
    T = tensor_from(U, V, W)
    for l in range(r):
        u = U[:, l].reshape(n, m)
        v = V[:, l].reshape(m, p)
        w = W[:, l].reshape(p, n)
        term = np.einsum('ij,jk,ki->ijk', u, v, w)
        if np.allclose(term, 0.0, atol=1e-12):
            return l
    return None


def try_merge_proportional(U, V, W):
    """Find two proportional columns and merge them. Returns (l1, l2, new_col) or None."""
    r = U.shape[1]
    for l1 in range(r):
        for l2 in range(l1 + 1, r):
            u1, u2 = U[:, l1], U[:, l2]
            v1, v2 = V[:, l1], V[:, l2]
            w1, w2 = W[:, l1], W[:, l2]
            # Check proportionality
            def scale(a, b):
                s = None
                for i in range(len(a)):
                    if abs(a[i]) > 1e-12:
                        s = b[i] / a[i]
                        break
                if s is None:
                    return 0.0 if np.allclose(b, 0, atol=1e-12) else None
                if not np.allclose(b, s * a, atol=1e-9):
                    return None
                return s
            su, sv, sw = scale(u1, u2), scale(v1, v2), scale(w1, w2)
            if su is not None and sv is not None and sw is not None:
                factor = 1 + su * sv * sw
                if abs(factor) < 1e-12:
                    # Sum is zero; drop both
                    return (l1, l2, None)
                new_u = factor * u1
                new_v = factor * v1
                new_w = factor * w1
                return (l1, l2, (new_u, new_v, new_w))
    return None


def try_eliminate(U, V, W):
    """Try to eliminate one column by expressing its tensor contribution as a linear
    combination of the others. Returns (index, alpha_vector) or None."""
    r = U.shape[1]
    T = tensor_from(U, V, W)

    # Precompute per-column tensor contributions for speed
    col_terms = []
    for l in range(r):
        u = U[:, l].reshape(n, m)
        v = V[:, l].reshape(m, p)
        w = W[:, l].reshape(p, n)
        col_terms.append(np.einsum('ij,jk,ki->ijk', u, v, w).flatten())

    for c in range(r):
        # Build system: col_terms[c] = sum_{l != c} alpha_l * col_terms[l]
        # Flatten: (n*m*p,) vector, unknowns: (r-1,)
        b = col_terms[c]
        # Build A matrix: each column l != c is col_terms[l]
        cols = []
        for l in range(r):
            if l != c:
                cols.append(col_terms[l])
        if not cols:
            continue
        A = np.column_stack(cols)
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except Exception:
            continue
        # Check residual
        recon = A @ alpha
        if np.max(np.abs(recon - b)) < 1e-9:
            return (c, alpha)
    return None


def reduce_rank(U, V, W, deadline):
    """Iteratively reduce rank using drop/merge/eliminate until stable or deadline."""
    while time.time() < deadline - 10:
        r = U.shape[1]
        if r <= 1:
            break

        # 1. Drop zero columns
        l = try_drop_zero(U, V, W)
        if l is not None:
            keep = [i for i in range(r) if i != l]
            U, V, W = U[:, keep], V[:, keep], W[:, keep]
            continue

        # 2. Merge proportional columns
        res = try_merge_proportional(U, V, W)
        if res is not None:
            l1, l2, new = res
            keep = [i for i in range(r) if i not in (l1, l2)]
            if new is not None:
                new_u, new_v, new_w = new
                U = np.column_stack([U[:, keep], new_u])
                V = np.column_stack([V[:, keep], new_v])
                W = np.column_stack([W[:, keep], new_w])
            else:
                U, V, W = U[:, keep], V[:, keep], W[:, keep]
            continue

        # 3. Eliminate linearly dependent columns
        res = try_eliminate(U, V, W)
        if res is not None:
            c, alpha = res
            keep = [i for i in range(r) if i != c]
            U, V, W = U[:, keep], V[:, keep], W[:, keep]
            continue

        # No more reductions
        break

    return U, V, W


def construct():
    deadline = time.time() + 138  # safety margin before 150s kill

    # Build baseline
    U_l, V_l, W_l = baseline_66()
    U = np.array(U_l, dtype=float)
    V = np.array(V_l, dtype=float)
    W = np.array(W_l, dtype=float)

    # Verify baseline
    if not verify(U, V, W):
        raise ValueError("Baseline rank-66 decomposition is invalid")

    # Reduce rank
    U, V, W = reduce_rank(U, V, W, deadline)

    # Final verification
    if not verify(U, V, W):
        # Fallback to baseline
        U = np.array(U_l, dtype=float)
        V = np.array(V_l, dtype=float)
        W = np.array(W_l, dtype=float)

    # Snap to half-integers if close (the task allows half-integer entries)
    # Only snap if it preserves exactness
    U_snap = np.round(2 * U) / 2.0
    V_snap = np.round(2 * V) / 2.0
    W_snap = np.round(2 * W) / 2.0
    if verify(U_snap, V_snap, W_snap, tol=1e-8):
        U, V, W = U_snap, V_snap, W_snap

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
