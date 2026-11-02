import numpy as np

DIMENSIONS = [2, 3, 7]


def _build_tensor(n, m, p):
    """T[i, j, k] = 1 if (i,j) is in A, (j,k) in B, (k,i) in C output index."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _round_to_halfint(a, max_den=8):
    """Round each entry to nearest multiple of 1/max_den, return float array."""
    return np.round(a * max_den) / max_den


def _verify(U, V, W, T, tol=1e-9):
    """Check sum_r U[:,r] V[:,r] W[:,r] == T."""
    r = U.shape[1]
    recon = np.tensordot(U, V, axes=(1, 1))  # (nm, mp)
    recon = np.tensordot(recon, W, axes=(1, 1))  # (nm, mp, pn)
    return bool(np.allclose(recon, T, atol=tol, rtol=0))


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    flat = T.reshape(-1)

    # Trivial valid fallback (always correct).
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0)); V0 = np.zeros((m * p, rank0)); W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    fallback = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Randomized rank-revealing: build tall matrix M (nm*mp x pn) via random
    # linear combinations to capture the rank of T.
    rng = np.random.default_rng(0)
    R = rng.standard_normal((p * n, 64))
    M = np.tensordot(T, R, axes=(2, 0))  # (nm, mp, 64) -> reshape
    M = M.reshape(n * m * m * p, 64)

    # SVD to estimate numerical rank.
    Umat, svals, Vt = np.linalg.svd(M, full_matrices=False)
    # Numerical rank: count singular values above tolerance.
    tol = svals[0] * max(M.shape) * 1e-12
    num_rank = int(np.sum(svals > tol))
    # Cap at trivial rank.
    rank = min(num_rank, rank0)
    rank = max(rank, 1)

    best = None
    # Try several random right-factors to extract a rank-`rank` decomposition.
    for trial in range(4):
        Q = rng.standard_normal((p * n, rank))
        # Solve least squares for M = X Q  ->  X = M Q^+
        Qp = np.linalg.pinv(Q)
        X = M @ Qp  # (nm*mp, rank)
        X = X.reshape(n * m, m * p, rank)
        # Now factor X[i,j,r] = sum? We need X = U[i,:] V[j,:] per r.
        # For each r, X[:,:,r] is nm x mp; factor via SVD to rank-1 sum.
        # Build U,V,W by concatenating rank-1 factors across r and i,j.
        # Simpler: treat each (r) slice as a matrix and use its SVD.
        U_list, V_list, W_list = [], [], []
        for r in range(rank):
            Mr = X[:, :, r]
            Ur, sr, Vr = np.linalg.svd(Mr, full_matrices=False)
            # Keep only components with non-negligible singular value.
            if len(sr) == 0:
                continue
            rtol = sr[0] * max(Mr.shape) * 1e-10
            keep = int(np.sum(sr > rtol))
            keep = max(keep, 1)
            for t in range(keep):
                U_list.append(Ur[:, t])
                V_list.append(Vr[:, t])
                W_list.append(np.full(p * n, 1.0) * sr[t])  # placeholder
        # The above W placeholder is not a valid decomposition in general.
        # Instead, do a proper CP-style extraction is complex; skip if not verified.
        if not U_list:
            continue
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        # Round to half-integers.
        U = _round_to_halfint(U, 8)
        V = _round_to_halfint(V, 8)
        W = _round_to_halfint(W, 8)
        if _verify(U, V, W, T):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break

    if best is not None:
        return best
    return fallback
