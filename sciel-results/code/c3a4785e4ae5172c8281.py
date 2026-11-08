"""Matrix multiplication 2x3x10 rank minimization via exact integer linear algebra."""
import time
import numpy as np
from math import gcd
from fractions import Fraction

DIMENSIONS = [2, 3, 10]


def _construct_rank_60():
    """Naive rank-60 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)


def _construct_rank_55():
    """Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W00: A02 * B2,k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W01: A02 * B2,k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # W10: A12 * B2,k0
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # W11: A12 * B2,k1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    return np.array(U, dtype=int), np.array(V, dtype=int), np.array(W, dtype=int)


def _smith_normal_form(M):
    """Compute Smith Normal Form of integer matrix M.
    Returns (D, P, Q) such that P @ M @ Q = D where D is diagonal.
    Uses fraction-free Gaussian elimination."""
    M = np.array(M, dtype=object)
    rows, cols = M.shape
    D = np.zeros((rows, cols), dtype=object)
    P = np.identity(rows, dtype=object)
    Q = np.identity(cols, dtype=object)

    for k in range(min(rows, cols)):
        # Find pivot: nonzero element at or below/right of (k,k)
        pivot = None
        for i in range(k, rows):
            for j in range(k, cols):
                if M[i, j] != 0:
                    pivot = (i, j)
                    break
            if pivot:
                break
        if pivot is None:
            break
        i, j = pivot
        # Swap rows
        if i != k:
            M[[k, i]] = M[[i, k]]
            P[[k, i]] = P[[i, k]]
        # Swap cols
        if j != k:
            M[:, [k, j]] = M[:, [j, k]]
            Q[[k, j]] = Q[[j, k]]

        # Make pivot positive
        if M[k, k] < 0:
            M[k, :] = -M[k, :]
            P[k, :] = -P[k, :]

        # Clear row k below pivot
        for ii in range(k + 1, rows):
            while M[ii, k] != 0:
                q = M[ii, k] // M[k, k]
                r = M[ii, k] - q * M[k, k]
                if abs(r) < abs(M[ii, k]):
                    M[ii, :] = M[ii, :] - q * M[k, :]
                    P[ii, :] = P[ii, :] - q * P[k, :]
                else:
                    # swap rows to reduce
                    M[[k, ii]] = M[[ii, k]]
                    P[[k, ii]] = P[[ii, k]]
                    if M[k, k] < 0:
                        M[k, :] = -M[k, :]
                        P[k, :] = -P[k, :]

        # Clear col k right of pivot
        for jj in range(k + 1, cols):
            while M[k, jj] != 0:
                q = M[k, jj] // M[k, k]
                r = M[k, jj] - q * M[k, k]
                if abs(r) < abs(M[k, jj]):
                    M[:, jj] = M[:, jj] - q * M[:, k]
                    Q[:, jj] = Q[:, jj] - q * Q[:, k]
                else:
                    M[:, [k, jj]] = M[:, [jj, k]]
                    Q[[k, jj]] = Q[[jj, k]]
                    if M[k, k] < 0:
                        M[k, :] = -M[k, :]
                        P[k, :] = -P[k, :]

        # Make pivot divide all entries in row and column
        changed = True
        while changed:
            changed = False
            for ii in range(rows):
                if M[ii, k] % M[k, k] != 0:
                    # Row operation
                    q = M[ii, k] // M[k, k]
                    M[ii, :] = M[ii, :] - q * M[k, :]
                    P[ii, :] = P[ii, :] - q * P[k, :]
                    changed = True
            for jj in range(cols):
                if M[k, jj] % M[k, k] != 0:
                    q = M[k, jj] // M[k, k]
                    M[:, jj] = M[:, jj] - q * M[:, k]
                    Q[:, jj] = Q[:, jj] - q * Q[:, k]
                    changed = True

    D = M.copy()
    return D, P, Q


def _find_kernel_vectors(M):
    """Find integer kernel vectors of M (solutions to M @ x = 0).
    Uses Smith Normal Form. Returns list of kernel vectors."""
    rows, cols = M.shape
    D, P, Q = _smith_normal_form(M)
    # D is diagonal with d_0, ..., d_{r-1} nonzero
    r = 0
    for i in range(min(rows, cols)):
        if D[i, i] != 0:
            r = i + 1
        else:
            break
    # Kernel dimension = cols - r
    # For each free variable, find a kernel vector
    kernel_vectors = []
    for free in range(r, cols):
        v = np.zeros(cols, dtype=object)
        v[free] = 1
        # Apply Q inverse (which is Q^T since Q is permutation-like)
        # Actually, kernel of M = Q @ kernel of D
        # Kernel of D: for i >= r, e_i
        # So kernel vector = Q[:, free]
        vec = np.array([Q[i, free] for i in range(cols)], dtype=object)
        kernel_vectors.append(vec)
    return kernel_vectors


def _check_exact_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    For each output element (i,k), check:
    sum_r U[i,r] * V[:,r] * W[k,r] matches the tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # The tensor T[i,j,k] = delta_{i,k} (for (A*B)_{i,k} = sum_j A_{i,j} B_{j,k})
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Check: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 if k==k else 0
    # Actually T[i,j,k] = 1 for all valid (i,j,k) since it's the identity in the middle
    # Wait, the tensor for matmul: T_{i,j,k} = 1 (it's just the multiplication)
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 for all i,j,k
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += int(U[i*m+j, r]) * int(V[j*p+k, r]) * int(W[k*n+i, r])
                if s != 1:
                    return False
    return True


def _drop_column_exact(U, V, W, col, keep_cols):
    """Try to drop column `col` by expressing W[:,col] as an integer combination
    of W[:,keep_cols]. Returns (new_U, new_V, new_W) or None."""
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    # Find integer a such that Wk @ a = Wc
    # Use Smith Normal Form on Wk
    try:
        D, P, Q = _smith_normal_form(Wk)
    except Exception:
        return None
    rows, ncols = Wk.shape
    # Check if Wc is in the column space of Wk
    # Wk @ a = Wc  <=>  D @ (Q^T @ a) = P @ Wc
    # Since D is diagonal, this is easy
    PWc = P @ Wc
    # For i < rank(D): D[i,i] * x_i = PWc[i], so x_i = PWc[i] / D[i,i]
    # For i >= rank: PWc[i] must be 0
    r = 0
    for i in range(min(rows, ncols)):
        if D[i, i] != 0:
            r = i + 1
        else:
            break
    # Check consistency
    for i in range(r, rows):
        if PWc[i] != 0:
            return None
    # Solve for x
    x = np.zeros(ncols, dtype=object)
    for i in range(r):
        if PWc[i] % D[i, i] != 0:
            return None
        x[i] = PWc[i] // D[i, i]
    # a = Q @ x
    a = Q @ x
    # Verify
    if not np.all(Wk @ a == Wc):
        return None
    # Build new matrices
    new_U = np.zeros((U.shape[0], len(keep_cols)), dtype=int)
    new_V = np.zeros((V.shape[0], len(keep_cols)), dtype=int)
    new_W = np.zeros((W.shape[0], len(keep_cols)), dtype=int)
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + int(a[li]) * U[:, col]
        new_V[:, li] = V[:, kc] + int(a[li]) * V[:, col]
        new_W[:, li] = W[:, kc] + int(a[li]) * W[:, col]
    return new_U, new_V, new_W


def _local_improve_exact(U, V, W, time_budget):
    """Attempt to reduce rank via exact integer null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column_exact(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W = res
                cur_cols = [c for c in cur_cols if c != col]
                improved = True
                break
    return cur_U, cur_V, cur_W, len(cur_cols)


def _search_rank_54():
    """Try to find a rank-54 decomposition by searching for one more dependency."""
    n, m, p = DIMENSIONS
    # Start from rank-55 and try all pairs of columns to see if one can be
    # expressed as a combination of others with integer coefficients
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    # Try dropping each column
    for col in range(rank):
        keep_cols = [c for c in range(rank) if c != col]
        res = _drop_column_exact(U, V, W, col, keep_cols)
        if res is not None:
            return res
    return None


def _try_rank_reduction(U, V, W, target_rank, time_budget):
    """Try to reduce to target_rank by iterative exact column dropping."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    while len(cur_cols) > target_rank:
        if time.time() - start > time_budget:
            break
        found = False
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            res = _drop_column_exact(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W = res
                cur_cols = [c for c in cur_cols if c != col]
                found = True
                break
        if not found:
            break
    return cur_U, cur_V, cur_W, len(cur_cols)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Start with rank-55
    U, V, W = _construct_rank_55()
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    # Try to reduce further with exact integer arithmetic
    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve_exact(U, V, W, time_budget)
            if new_rank < best[0]:
                best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    # Try to search for rank 54
    if time.time() < DEADLINE - 10:
        try:
            res = _search_rank_54()
            if res is not None:
                nU, nV, nW = res
                if nU.shape[1] < best[0]:
                    best = (nU.shape[1], nU, nV, nW)
        except Exception:
            pass

    # Try aggressive reduction from best
    if time.time() < DEADLINE - 5:
        r, bU, bV, bW = best
        if r > 50:
            time_budget = max(1.0, DEADLINE - time.time() - 5.0)
            try:
                nU, nV, nW, new_rank = _try_rank_reduction(bU, bV, bW, 50, time_budget)
                if new_rank < best[0]:
                    best = (new_rank, nU, nV, nW)
            except Exception:
                pass

    r, U, V, W = best
    # Convert to lists
    U = U.astype(int).tolist()
    V = V.astype(int).tolist()
    W = W.astype(int).tolist()
    return {"U": U, "V": V, "W": W}
