import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W

def _build_true_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    return T

def _max_residual(U, V, W, T_true):
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(M - T_true))

def _try_eliminate_column(U0, V0, W0, T_true, target_col, time_budget):
    """Use smooth epigraph optimization to drive column target_col to zero
    while keeping the decomposition accurate."""
    start = time.time()
    nm, mp, pn = U0.shape[0], V0.shape[0], W0.shape[0]
    rank = U0.shape[1]

    # Flatten parameters: U, V, W, and t (epigraph variable)
    def params_to_mats(x):
        nU = nm * rank
        nV = mp * rank
        nW = pn * rank
        U = x[:nU].reshape(nm, rank)
        V = x[nU:nU+nV].reshape(mp, rank)
        W = x[nU+nV:nU+nV+nW].reshape(pn, rank)
        t = x[nU+nV+nW]
        return U, V, W, t

    def objective(x):
        U, V, W, t = params_to_mats(x)
        # Residual
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        resid = M - T_true
        # Soft max of |resid| via log-sum-exp (smooth approximation to max)
        # Use a large enough beta for tightness
        beta = 50.0
        # For numerical stability
        r2 = resid ** 2
        max_r2 = np.max(r2)
        if max_r2 < 1e-30:
            soft_max = 0.0
        else:
            # log-sum-exp of r2, then sqrt
            shifted = r2 - max_r2
            lse = max_r2 + np.log(np.sum(np.exp(shifted)))
            soft_max = np.sqrt(lse)
        # Regularization to push target column to zero
        col_norm_sq = (U[:, target_col] ** 2).sum() + (V[:, target_col] ** 2).sum() + (W[:, target_col] ** 2).sum()
        lam = 5.0
        # Epigraph: t should be >= soft_max residual
        # Minimize t + lam * col_norm_sq + penalty for t < soft_max
        penalty = 0.0
        if t < soft_max:
            penalty = 100.0 * (soft_max - t) ** 2
        return t + lam * col_norm_sq + penalty

    def gradient(x):
        U, V, W, t = params_to_mats(x)
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        resid = M - T_true
        beta = 50.0
        r2 = resid ** 2
        max_r2 = np.max(r2)
        if max_r2 < 1e-30:
            # Gradient is zero for the soft-max part
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
        else:
            shifted = r2 - max_r2
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            # d(soft_max)/d(resid) = weights * 2 * resid / soft_max
            soft_max = np.sqrt(max_r2 + np.log(np.sum(np.exp(shifted))))
            if soft_max < 1e-15:
                d_resid = np.zeros_like(resid)
            else:
                d_resid = 2.0 * weights * resid / soft_max
            # dU[i, r] = sum_{j,k} d_resid[i,j,k] * V[j,r] * W[k,r]
            dU = np.einsum('ijk,jr,kr->ir', d_resid, V, W)
            dV = np.einsum('ijk,ir,kr->jr', d_resid, U, W)
            dW = np.einsum('ijk,ir,jr->kr', d_resid, U, V)

        # Gradient of regularization
        lam = 5.0
        dU[:, target_col] += 2.0 * lam * U[:, target_col]
        dV[:, target_col] += 2.0 * lam * V[:, target_col]
        dW[:, target_col] += 2.0 * lam * W[:, target_col]

        # Gradient of penalty
        dt = 1.0
        if t < soft_max:
            dt += 200.0 * (t - soft_max)

        nU = nm * rank
        nV = mp * rank
        nW = pn * rank
        gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten(), [dt]])
        return gx

    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten(), [1.0]])

    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    except Exception:
        return None, None, None, None

    if time.time() - start > time_budget:
        return None, None, None, None

    U_opt, V_opt, W_opt, t_opt = params_to_mats(res.x)
    col_norm = np.sqrt((U_opt[:, target_col] ** 2).sum() + (V_opt[:, target_col] ** 2).sum() + (W_opt[:, target_col] ** 2).sum())

    # Check if the column is effectively zero
    if col_norm < 1e-6:
        # Verify the remaining 31 columns give an exact decomposition
        U31 = np.delete(U_opt, target_col, axis=1)
        V31 = np.delete(V_opt, target_col, axis=1)
        W31 = np.delete(W_opt, target_col, axis=1)
        M31 = np.einsum('ir,jr,kr->ijk', U31, V31, W31)
        max_res = np.max(np.abs(M31 - T_true))
        if max_res < 1e-8:
            return U31, V31, W31, max_res

    return None, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Build the guaranteed rank-32 solution
    U32, V32, W32 = _build_rank32()
    T_true = _build_true_tensor(n, m, p)

    # Verify the rank-32 solution is correct
    assert _max_residual(U32, V32, W32, T_true) < 1e-10, "Rank-32 solution is incorrect"

    best_U, best_V, best_W = U32.tolist(), V32.tolist(), W32.tolist()
    best_rank = 32

    # Try to find a rank-31 decomposition by eliminating each column
    rank = 32
    time_per_col = 25.0  # Allow up to 25s per column attempt, but check global deadline

    for target_col in range(rank):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        U31, V31, W31, max_res = _try_eliminate_column(
            U32, V32, W32, T_true, target_col, min(time_per_col, remaining - 5.0)
        )

        if U31 is not None and max_res < 1e-8:
            best_U = U31.tolist()
            best_V = V31.tolist()
            best_W = W31.tolist()
            best_rank = 31
            break  # Found rank-31, stop searching

    return {"U": best_U, "V": best_V, "W": best_W}
