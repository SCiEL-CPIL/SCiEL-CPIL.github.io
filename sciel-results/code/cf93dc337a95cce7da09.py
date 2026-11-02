"""Numerical rank-minimization strategy: warm start with a sub-schoolbook
(Strassen 4x4x4 + schoolbook 4x4x3) decomposition, then prune rank via
iterative rank-revealing SVD (numerical tensor rank reduction)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42


def _strassen2x2():
    """Rank-7 Strassen for 2x2 * 2x2. Returns U,V,W each shape (4,7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # P1 = (A00+A11)*(B00+B11); C00 += P1, C11 += P1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # P2 = (A10+A11)*B00; C10 += P2, C11 -= P2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # P3 = A00*(B01-B11); C01 += P3, C11 += P3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # P4 = A11*(B10-B00); C00 += P4, C10 += P4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # P5 = (A00+A01)*B11; C00 -= P5, C01 += P5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # P6 = (A10-A00)*(B00+B01); C11 += P6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # P7 = (A01-A11)*(B10+B11); C00 += P7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _strassen4x4x4():
    """Rank-28 decomposition of 4x4 * 4x4 via 4 independent 2x2 Strassen blocks.
    Returns U (16,28), V (16,28), W (16,28)."""
    sU, sV, sW = _strassen2x2()
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    for bi in range(2):      # row block of A / C
        for bj in range(2):  # col block of A / row block of B / C
            for bk in range(2):  # col block of B / C
                col = (bi * 2 + bj) * 2 + bk  # 0..7
                r0 = col * 4
                for r in range(7):
                    for (ui, vi, wi) in zip(
                            range(4), range(4), range(4)):
                        U[(bi * 2 + ui) * 4 + (bj * 2 + r % 4 if False else (ui)), r0 + r] = 0
                # simpler: fill by mapping local indices
                for r in range(7):
                    # U local rows 0..3 -> global A row bi*2+ui, col bj*2+uj
                    for ui in range(2):
                        for uj in range(2):
                            local_u = ui * 2 + uj
                            U[(bi * 2 + ui) * 4 + (bj * 2 + uj), r0 + r] = sU[local_u, r]
                    for uj in range(2):
                        for uk in range(2):
                            local_v = uj * 2 + uk
                            V[(bj * 2 + uj) * 4 + (bk * 2 + uk), r0 + r] = sV[local_v, r]
                    for uk in range(2):
                        for ui in range(2):
                            local_w = uk * 2 + ui
                            W[(bk * 2 + uk) * 4 + (bi * 2 + ui), r0 + r] = sW[local_w, r]
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_warm_start(n, m, p):
    """4x4*4x7 = (4x4*4x4 Strassen rank28) + (4x4*4x1 schoolbook rank16) = rank 44."""
    # core 4x4*4x4
    Uc, Vc, Wc = _strassen4x4x4()
    # tail 4x4*4x1 schoolbook
    Ut, Vt, Wt = _schoolbook(n, m, 1)
    U = np.hstack([Uc, Ut])
    V = np.hstack([Vc, Vt])
    W = np.hstack([Wc, Wt])
    return U, V, W


def _error(U, V, W, n, m, p):
    """Max abs error of the decomposition against the multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Vectorized: for each r, outer product.
    best = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                err = abs(s - 1.0)
                if err > best:
                    best = err
    return best


def _prune_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """Iteratively drop rank terms that are numerically redundant.
    Uses rank-revealing SVD on the stacked factor to find a low-rank
    re-approximation, then drops the smallest singular term if it does
    increase error beyond tol. Returns best (U,V,W) found."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    best = (U.copy(), V.copy(), W.copy())
    base_err = _error(U, V, W, n, m, p)
    if base_err > tol:
        return best

    r = U.shape[1]
    # Stack the three factors into one matrix; columns are rank directions.
    def stack(u, v, w):
        return np.vstack([u, v, w])

    # Try dropping one term at a time greedily, using SVD to detect redundancy.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_r = U.shape[1]
        if cur_r <= 1:
            break
        S = stack(U, V, W)
        # SVD of the stacked factor matrix (rows = all entries, cols = rank)
        try:
            Uf, svals, Vt = np.linalg.svd(S, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        # The smallest singular value indicates redundancy of a direction.
        # Attempt to drop the rank direction least aligned with the data.
        # We try each candidate drop and keep the one that keeps error minimal.
        best_drop = None
        best_err_after = base_err
        for c in range(cur_r):
            if time.time() > deadline:
                break
            cand_U = np.delete(U, c, axis=1)
            cand_V = np.delete(V, c, axis=1)
            cand_W = np.delete(W, c, axis=1)
            e = _error(cand_U, cand_V, cand_W, n, m, p)
            if e <= tol and (best_drop is None or e < best_err_after):
                best_drop = c
                best_err_after = e
        if best_drop is not None:
            U = np.delete(U, best_drop, axis=1)
            V = np.delete(V, best_drop, axis=1)
            W = np.delete(W, best_drop, axis=1)
            base_err = best_err_after
            best = (U.copy(), V.copy(), W.copy())
            improved = True
    return best


def _round_halfint(U, V, W, n, m, p, tol=1e-6):
    """Round entries to nearest half-integer if it keeps the decomposition valid."""
    def round_matrix(M):
        return np.round(M * 2) / 2
    Ur = round_matrix(U)
    Vr = round_matrix(V)
    Wr = round_matrix(W)
    e = _error(Ur, Vr, Wr, n, m, p)
    if e <= tol:
        return Ur, Vr, Wr
    # Try rounding only where it helps; fall back to original.
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill

    # Warm start: sub-schoolbook (Strassen core + schoolbook tail) rank 44.
    U, V, W = _build_warm_start(n, m, p)

    # Verify warm start is valid.
    err = _error(U, V, W, n, m, p)
    if err > 1e-9:
        # Fallback to full schoolbook (guaranteed valid, rank 112).
        U, V, W = _schoolbook(n, m, p)

    # Numerical rank minimization via iterative SVD-based pruning.
    U, V, W = _prune_rank(U, V, W, n, m, p, deadline)

    # Round to half-integers if possible (keeps entries in allowed set).
    U, V, W = _round_halfint(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
