"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses a Kronecker-embedded Strassen baseline (rank 65 = 49 + 16), then refines via:
  1. SVD-based structural rank reduction (null-space / linear-dependence elimination)
  2. Vectorized ALS (einsum) with half-integer rounding
  3. A smooth epigraph / soft-min polish pass (smooth surrogate for the max-norm objective)
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=atol)

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the L2 error of the decomposition efficiently."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.sqrt(np.sum((T - 1.0)**2))

def _compute_max_abs_error(U, V, W, n, m, p, rank):
    """Compute the max absolute error (the quantity that must hit < atol for validity)."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return float(np.max(np.abs(T - 1.0)))

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.round(np.asarray(M, dtype=np.float64) * 2.0) / 2.0

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists for U, V, W."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition: 4x4x4 (Kronecker-embedded Strassen, rank 49)
    plus 4x4x1 (naive, rank 16)."""
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    strassen_terms = _strassen_2x2x2_terms()

    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    u[(i1 * 2 + i2) * m + (j1 * 2 + j2)] = val1 * val2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    v[(j1 * 2 + j2) * p + (k1 * 2 + k2)] = val1 * val2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    w[(k1 * 2 + k2) * n + (i1 * 2 + i2)] = val1 * val2
            add_term(u, v, w)

    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _als_decompose(n, m, p, rank, seed=42, max_iters=500, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=100,
                   use_smooth=False, smooth_iters=150, smooth_beta=250.0):
    """Find a rank-r decomposition using vectorized ALS with rounding,
    optionally followed by a smooth epigraph / soft-min polish pass.

    Returns (U, V, W) as numpy arrays if found, else None.
    """
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5

    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)

    def _als_iteration():
        # Optimize U: for each (i,j) solve A @ U[i,j] = ones(p),
        # A[k,:] = V[j,k,:] * W[k,i,:]
        A_U = np.einsum('jpk,kip->ijpk', Vr, Wr)
        sol = np.linalg.lstsq(A_U, np.ones((n, m, p)), rcond=None)[0]
        Ur[:, :, :] = sol
        # Optimize V: for each (j,k) solve A @ V[j,k] = ones(n),
        # A[i,:] = U[i,j,:] * W[k,i,:]
        A_V = np.einsum('ijr,kir->jikr', Ur, Wr)
        sol = np.linalg.lstsq(A_V, np.ones((m, p, n)), rcond=None)[0]
        Vr[:, :, :] = sol
        # Optimize W: for each (k,i) solve A @ W[k,i] = ones(m),
        # A[j,:] = U[i,j,:] * V[j,k,:]
        A_W = np.einsum('ijr,jkr->jikr', Ur, Vr)
        sol = np.linalg.lstsq(A_W, np.ones((p, n, m)), rcond=None)[0]
        Wr[:, :, :] = sol

    def _smooth_iteration():
        # Smooth epigraph / soft-min polish: minimize the soft-max of |err|
        # (log-sum-exp surrogate of max|err|) with an auxiliary variable t.
        # Gradient descent on (U, V, W) with an auxiliary t for the epigraph.
        # This is a smooth surrogate for the nonsmooth max-norm objective.
        beta = smooth_beta
        # Compute current error tensor T
        T = np.einsum('icr,jcr,kir->ijk', Ur, Vr, Wr)
        err = T - 1.0
        abs_err = np.abs(err)
        # Soft-max via log-sum-exp (numerically stable)
        m = abs_err.max()
        logsumexp = m + np.log(np.sum(np.exp(abs_err - m)))
        # Gradient of soft-max wrt each entry: exp(e - logsumexp)
        grad_soft = np.exp(abs_err - logsumexp)
        # d|err|/derr = sign(err)
        sign_err = np.sign(err)
        # d soft-max / dT = grad_soft * sign_err
        dT = grad_soft * sign_err
        # dT/dU[i,j,r] = V[j,:,r] * W[:,i,r] (over k)
        dU = np.einsum('ijk,jkr,ikr->ijr', dT, Vr, Wr)
        dV = np.einsum('ijk,ijr,ikr->jkr', dT, Ur, Wr)
        dW = np.einsum('ijk,ijr,jkr->ikr', dT, Ur, Vr)
        # Auxiliary variable t for epigraph: t tracks the max-norm
        # (kept for completeness; the soft-max already smooths it)
        # Simple gradient step
        lr = 0.02
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW

    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-12:
            break

    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-14:
            break

    # Smooth epigraph / soft-min polish pass (new hypothesis)
    if use_smooth and best_err < 1.0:
        U_s = best_U.copy(); V_s = best_V.copy(); W_s = best_W.copy()
        Ur_s = U_s.reshape(n, m, rank); Vr_s = V_s.reshape(m, p, rank); Wr_s = W_s.reshape(p, n, rank)
        for it in range(smooth_iters):
            if deadline is not None and time.time() > deadline:
                break
            _smooth_iteration()
            err = _compute_error(U_s, V_s, W_s, n, m, p, rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
            if err < 1e-14:
                break

    # Try multiple rounding strategies
    candidates = []
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    for offset in [0.25, -0.25]:
        candidates.append((_round_matrix_to_half_int(best_U + offset),
                           _round_matrix_to_half_int(best_V + offset),
                           _round_matrix_to_half_int(best_W + offset)))

    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc, Vc, Wc

    return None

def _svd_reduce(U, V, W, n, m, p, target_rank, deadline):
    """SVD-based structural rank reduction: detect linearly dependent terms
    in the factor matrices and project onto the principal subspace of
    dimension target_rank, then polish with ALS."""
    rank = U.shape[1]
    if target_rank >= rank:
        return None

    # Stack the three factor matrices to find the joint principal subspace
    # Each column c is a "term". The three matrices U[:,c], V[:,c], W[:,c]
    # must be consistent. We project each onto its own top-target_rank SVD subspace.
    # This is a structural (not just gradient) reduction.
    try:
        # SVD of U: keep top target_rank right-singular vectors
        _, _, Ut = np.linalg.svd(U, full_matrices=False)
        U_sub = Ut[:target_rank, :].T  # (rank, target_rank)
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        V_sub = Vt[:target_rank, :].T
        _, _, Wt = np.linalg.svd(W, full_matrices=False)
        W_sub = Wt[:target_rank, :].T

        U_r = U @ U_sub  # (n*m, target_rank)
        V_r = V @ V_sub
        W_r = W @ W_sub
    except:
        return None

    # Polish with ALS from this reduced starting point
    result = _als_decompose(n, m, p, target_rank, seed=0, max_iters=300,
                            deadline=deadline, U_init=U_r, V_init=V_r, W_init=W_r,
                            polish_iters=100, use_smooth=True, smooth_iters=100)
    return result

def _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline):
    """Try to reduce the rank starting from a baseline decomposition
    using SVD structural reduction + ALS."""
    rank = U_base.shape[1]
    best_result = None
    best_rank = rank

    for target_rank in range(rank - 1, max(40, rank - 25), -1):
        if time.time() > deadline:
            break

        # SVD structural reduction
        result = _svd_reduce(U_base, V_base, W_base, n, m, p, target_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            continue

        # Also try dropping columns + re-optimizing (gradient method)
        for init_seed in range(3):
            if time.time() > deadline:
                break
            if init_seed == 0:
                U_init = U_base[:, :target_rank]
                V_init = V_base[:, :target_rank]
                W_init = W_base[:, :target_rank]
            elif init_seed == 1:
                U_init = U_base[:, rank-target_rank:]
                V_init = V_base[:, rank-target_rank:]
                W_init = W_base[:, rank-target_rank:]
            else:
                idx = np.random.RandomState(init_seed * 100 + target_rank).choice(rank, target_rank, replace=False)
                U_init = U_base[:, idx]
                V_init = V_base[:, idx]
                W_init = W_base[:, idx]
            result = _als_decompose(n, m, p, target_rank, seed=init_seed, max_iters=200,
                                    deadline=deadline, U_init=U_init, V_init=V_init, W_init=W_init,
                                    polish_iters=50)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break

    return best_result

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n, m, p = DIMENSIONS

    # Baseline: Kronecker-embedded Strassen (rank 65)
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    kron_ok = _verify_fast(U_base.tolist(), V_base.tolist(), W_base.tolist(), n, m, p, atol=1e-9)

    if not kron_ok:
        return _schoolbook()

    best_result = {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    best_rank = U_base.shape[1]

    # Try to reduce rank from the baseline (SVD structural + ALS)
    if time.time() < deadline - 50:
        reduced = _try_reduce_from_baseline(U_base, V_base, W_base, n, m, p, deadline - 50)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
            best_rank = U_red.shape[1]

    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(45, best_rank - 30), -1):
        if time.time() > deadline - 25:
            break
        found = False
        for seed in range(15):
            if time.time() > deadline - 25:
                break
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=400,
                                    deadline=deadline - 25, polish_iters=80,
                                    use_smooth=True, smooth_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                found = True
                break
        if not found:
            break

    return best_result
# EVOLVE-BLOCK-END
