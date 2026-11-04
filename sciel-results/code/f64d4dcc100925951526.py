import numpy as np
import time
from scipy.optimize import least_squares

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
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

    def try_optimize_rank(R, max_restarts=8):
        """
        Try to find a decomposition of rank R using continuous optimization
        with multiple random restarts and a final polishing pass.
        """
        if time.time() > deadline - 25:
            return None

        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W

        # Precompute index arrays for faster residual evaluation
        u_indices, v_indices, w_indices = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_indices.append(i * m + j)
                    v_indices.append(j * p + k)
                    w_indices.append(k * n + i)
        u_indices = np.array(u_indices)
        v_indices = np.array(v_indices)
        w_indices = np.array(w_indices)

        def residuals(x):
            U = x[:size_U].reshape((n*m, R))
            V = x[size_U:size_U+size_V].reshape((m*p, R))
            W = x[size_U+size_V:].reshape((p*n, R))

            # Compute all trilinear forms at once using broadcasting
            # For each term (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is equivalent to: for each r, sum over (i,j,k) of U[u,r]*V[v,r]*W[w,r]
            # Let's compute per-term:
            # Reshape for vectorized computation
            U_sel = U[u_indices, :]  # shape (nmp, R)
            V_sel = V[v_indices, :]  # shape (nmp, R)
            W_sel = W[w_indices, :]  # shape (nmp, R)
            res = np.sum(U_sel * V_sel * W_sel, axis=1) - 1.0
            return res

        def try_initialization(x0):
            if time.time() > deadline - 20:
                return None

            try:
                # First pass: moderate tolerance
                result = least_squares(residuals, x0, method='lm', max_nfev=300,
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10)
            except Exception:
                return None

            if result.cost < 1e-6:
                # Second pass: polish with tighter tolerance
                try:
                    result2 = least_squares(residuals, result.x, method='lm', max_nfev=200,
                                            ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    if result2.cost < result.cost:
                        result = result2
                except Exception:
                    pass

            if result.cost < 1e-10:
                U = result.x[:size_U].reshape((n*m, R))
                V = result.x[size_U:size_U+size_V].reshape((m*p, R))
                W = result.x[size_U+size_V:].reshape((p*n, R))

                # Try to snap to integers/half-integers
                for scale in [2, 1]:
                    U_snap = np.round(U * scale) / scale
                    V_snap = np.round(V * scale) / scale
                    W_snap = np.round(W * scale) / scale

                    if check_decomposition(U_snap, V_snap, W_snap):
                        return (U_snap, V_snap, W_snap)

                # If snapping fails, try a final high-precision solve
                # using the snapped values as a new starting point
                x_snap = np.concatenate([
                    (np.round(U * 2) / 2).flatten(),
                    (np.round(V * 2) / 2).flatten(),
                    (np.round(W * 2) / 2).flatten()
                ])
                try:
                    result3 = least_squares(residuals, x_snap, method='lm', max_nfev=100,
                                            ftol=1e-15, xtol=1e-15, gtol=1e-15)
                    if result3.cost < 1e-12:
                        U_f = result3.x[:size_U].reshape((n*m, R))
                        V_f = result3.x[size_U:size_U+size_V].reshape((m*p, R))
                        W_f = result3.x[size_U+size_V:].reshape((p*n, R))
                        for scale in [2, 1]:
                            U_s = np.round(U_f * scale) / scale
                            V_s = np.round(V_f * scale) / scale
                            W_s = np.round(W_f * scale) / scale
                            if check_decomposition(U_s, V_s, W_s):
                                return (U_s, V_s, W_s)
                except Exception:
                    pass

        # Generate multiple initializations
        for attempt in range(max_restarts):
            if time.time() > deadline - 20:
                break

            # Mix of random and structured initializations
            if attempt == 0:
                # Near-schoolbook initialization (perturbed)
                U_s, V_s, W_s = get_schoolbook()
                if R <= 60:
                    U_init = U_s[:, :R].copy()
                    V_init = V_s[:, :R].copy()
                    W_init = W_s[:, :R].copy()
                    # Add small random perturbation to escape local minima
                    U_init += np.random.randn(n*m, R) * 0.1
                    V_init += np.random.randn(m*p, R) * 0.1
                    W_init += np.random.randn(p*n, R) * 0.1
                else:
                    U_init = np.random.randn(n*m, R) * 0.5
                    V_init = np.random.randn(m*p, R) * 0.5
                    W_init = np.random.randn(p*n, R) * 0.5
            elif attempt == 1:
                # Structured: try to use rank-1 components
                U_init = np.random.randn(n*m, R) * 0.3
                V_init = np.random.randn(m*p, R) * 0.3
                W_init = np.random.randn(p*n, R) * 0.3
            else:
                # Pure random
                U_init = np.random.randn(n*m, R) * 0.5
                V_init = np.random.randn(m*p, R) * 0.5
                W_init = np.random.randn(p*n, R) * 0.5

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = try_initialization(x0)
            if result is not None:
                return result

        return None

    # Try ranks from 59 down to 45
    for R in range(59, 44, -1):
        if time.time() > deadline - 30:
            break
        result = try_optimize_rank(R, max_restarts=6 if R > 55 else 4)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
