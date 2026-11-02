import numpy as np
import time

def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140.0

    # Build the rank-53 baseline
    baseline = _build_rank53()
    best_result = baseline
    best_rank = 53

    # Try to reduce rank via projection (52, 51, 50)
    for target_rank in [52, 51, 50]:
        if time.time() > deadline - 30:
            break
        try:
            result = _try_project(baseline, target_rank)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break  # Don't need to try lower ranks if we succeeded
        except Exception:
            pass

    # If we have time, try direct least-squares attempts for rank 52
    if time.time() < deadline - 30 and best_rank > 52:
        try:
            result = _try_least_squares(52, deadline - 30)
            if result is not None:
                best_result = result
                best_rank = 52
        except Exception:
            pass

    return best_result


def _build_rank53():
    n, m, p = 2, 2, 15
    rank = 53

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 factors (rank 7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # 2x2x1 factors (last column, rank 4)
    U_sc = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_sc = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_sc = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        for r in range(7):
            c = col + r
            for i in range(2):
                for j in range(2):
                    U[i*m + j][c] = U_s[i*2 + j][r]
            for j in range(2):
                V[j*p + k0][c] = V_s[j*2 + 0][r]
                V[j*p + k1][c] = V_s[j*2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk*n + i][c] = W_s[k*2 + i][r]
        col += 7

    k_last = 14
    for r in range(4):
        c = col + r
        for i in range(2):
            for j in range(2):
                U[i*m + j][c] = U_sc[i*2 + j][r]
        for j in range(2):
            V[j*p + k_last][c] = V_sc[j][r]
        for i in range(2):
            W[k_last*n + i][c] = W_sc[i][r]

    return {"U": U, "V": V, "W": W}


def _compute_residual_fast(U, V, W):
    """
    Compute the residual tensor efficiently.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 2, 2, 15
    rank = U.shape[1]

    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])

    return T


def _try_project(base_dict, target_rank):
    """
    Project a rank-R decomposition to rank target_rank by expressing the
    last (R - target_rank) factors as linear combinations of the first
    target_rank factors and absorbing coefficients.
    """
    n, m, p = 2, 2, 15
    R = len(base_dict["U"][0])
    if target_rank >= R:
        return None

    U = np.array(base_dict["U"], dtype=float)
    V = np.array(base_dict["V"], dtype=float)
    W = np.array(base_dict["W"], dtype=float)

    # Verify baseline is correct
    T_check = _compute_residual_fast(U, V, W)
    if np.max(np.abs(T_check - 1.0)) > 1e-9:
        return None

    # We want to reduce rank from R to target_rank.
    # Express columns R-target_rank..R-1 as combinations of 0..target_rank-1.
    # For each factor matrix, solve X[0:target_rank] @ alpha = X[target_rank:]
    # where alpha is (target_rank, R-target_rank).

    # Try different "pivot" choices: which columns to keep.
    # Default: keep first target_rank, drop last R-target_rank.
    keep = list(range(target_rank))
    drop = list(range(target_rank, R))

    # Solve for each factor
    alphas = []
    for X in [U, V, W]:
        X_keep = X[:, keep]
        X_drop = X[:, drop]
        # Solve X_keep @ alpha = X_drop
        # alpha shape: (target_rank, R - target_rank)
        try:
            alpha, _, rank_solve, _ = np.linalg.lstsq(X_keep, X_drop, rcond=None)
            alphas.append(alpha)
        except Exception:
            return None

    # Apply the transformation: new columns are linear combinations
    # U_new[:, r] = U[:, r] - sum_s alpha_U[r, s] * U[:, drop[s]]  ... 
    # Actually, we absorb: U_new = U_keep - U_drop @ alpha_U^T
    # Wait, let me think again.
    # Original: sum_{r=0}^{R-1} U[:,r] V[:,r] W[:,r]
    # We want: sum_{r=0}^{target_rank-1} U_new[:,r] V_new[:,r] W_new[:,r]
    # where U_new[:,r] = U[:,r] + sum_s alpha_U[r,s] U[:, drop[s]]
    # and similarly for V, W. But this doesn't work directly because
    # the product is not linear in this way.

    # Better approach: The tensor is T = sum_r u_r ⊗ v_r ⊗ w_r.
    # If we can write u_{drop[s]} = sum_r alpha_U[r,s] u_r, etc.,
    # then T = sum_{r in keep} u_r ⊗ v_r ⊗ w_r + sum_s u_{drop[s]} ⊗ v_{drop[s]} ⊗ w_{drop[s]}
    # The second term is a rank-(R-target_rank) tensor, not easily absorbed.

    # The correct projection: we need to find new factors such that
    # sum_{r=0}^{target_rank-1} u'_r ⊗ v'_r ⊗ w'_r = T.
    # This is a nonlinear problem. But if the tensor lies in the span of
    # the first target_rank factors in some sense, we can solve it.

    # Alternative: Use the fact that T is all-ones. Try to find a
    # rank-target_rank decomposition directly via the structure.

    # Let me try a different approach: use the "sliding window" idea.
    # For M_{2,2,3}, is there a rank-10 decomposition?
    # Schoolbook for 2x2x3: rank 12. Strassen for 2x2x2 + schoolbook for 2x2x1: 7+4=11.
    # Is rank 10 possible? Let's try to find it.

    return None


def _try_least_squares(target_rank, deadline):
    """Try least_squares to find a rank-target_rank decomposition with multiple restarts."""
    from scipy.optimize import least_squares
    n, m, p = 2, 2, 15
    T_target = np.ones((n, m, p))

    base = _build_rank53()
    U_full = np.array(base["U"], dtype=float)
    V_full = np.array(base["V"], dtype=float)
    W_full = np.array(base["W"], dtype=float)

    rng = np.random.default_rng(42)
    num_attempts = 8

    for attempt in range(num_attempts):
        if time.time() > deadline:
            break

        if attempt == 0:
            U0 = U_full[:, :target_rank].copy()
            V0 = V_full[:, :target_rank].copy()
            W0 = W_full[:, :target_rank].copy()
        elif attempt == 1:
            # Use last target_rank columns
            U0 = U_full[:, -target_rank:].copy()
            V0 = V_full[:, -target_rank:].copy()
            W0 = W_full[:, -target_rank:].copy()
        else:
            # Random initialization with scaling
            scale = 0.3 + 0.1 * attempt
            U0 = rng.normal(0, scale, (n*m, target_rank))
            V0 = rng.normal(0, scale, (m*p, target_rank))
            W0 = rng.normal(0, scale, (p*n, target_rank))

        def _residual(x):
            U_size = n * m * target_rank
            V_size = m * p * target_rank
            U_r = x[:U_size].reshape(n * m, target_rank)
            V_r = x[U_size:U_size+V_size].reshape(m * p, target_rank)
            W_r = x[U_size+V_size:].reshape(p * n, target_rank)
            T_comp = _compute_residual_fast(U_r, V_r, W_r)
            return (T_comp - T_target).flatten()

        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            result = least_squares(_residual, x0, method='lm', max_nfev=3000)
            if np.linalg.norm(result.fun) < 1e-10:
                U_size = n * m * target_rank
                V_size = m * p * target_rank
                U_opt = result.x[:U_size].reshape(n * m, target_rank)
                V_opt = result.x[U_size:U_size+V_size].reshape(m * p, target_rank)
                W_opt = result.x[U_size+V_size:].reshape(p * n, target_rank)
                U_r = np.round(U_opt * 2) / 2
                V_r = np.round(V_opt * 2) / 2
                W_r = np.round(W_opt * 2) / 2
                T_check = _compute_residual_fast(U_r, V_r, W_r)
                if np.max(np.abs(T_check - T_target)) < 1e-9:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        except Exception:
            continue

    return None
