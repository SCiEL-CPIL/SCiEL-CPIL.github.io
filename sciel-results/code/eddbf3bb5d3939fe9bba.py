import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def time_left():
        return DEADLINE_S - (time.time() - start_time)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_mat = U[:, c].reshape(n, m)
            v_mat = V[:, c].reshape(m, p)
            w_mat = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T_computed[i, j, :] += u_mat[i, j] * (v_mat[j, :] * w_mat[:, i])
        return np.max(np.abs(T_computed - 1.0)) < tol

    def get_17_rank_block():
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))

        col = 0
        # Strassen part (7 terms) for 2x2x2
        U_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: 1},
            {0: 1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 0: -1},
            {1: 1, 3: -1}
        ]
        V_coeffs = [
            {0: 1, 3: 1},
            {0: 1},
            {1: 1, 3: -1},
            {2: 1, 0: -1},
            {3: 1},
            {0: 1, 1: 1},
            {2: 1, 3: 1}
        ]
        W_coeffs = [
            {0: 1, 3: 1},
            {2: 1, 3: -1},
            {1: 1, 3: 1},
            {0: 1, 2: 1},
            {0: -1, 1: 1},
            {3: 1},
            {0: 1}
        ]

        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local * 3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local * 2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local * 3 + i_local, col] += coeff
            col += 1

        # Schoolbook part (10 terms)
        for i in range(2):
            for k in range(2):
                U[i * 3 + 2, col] = 1.0
                V[2 * 2 + k, col] = 1.0
                W[k * 3 + i, col] = 1.0
                col += 1
        for j in range(2):
            for k in range(2):
                U[2 * 3 + j, col] = 1.0
                V[j * 2 + k, col] = 1.0
                W[k * 3 + 2, col] = 1.0
                col += 1
        for k in range(2):
            U[2 * 3 + 2, col] = 1.0
            V[2 * 2 + k, col] = 1.0
            W[k * 3 + 2, col] = 1.0
            col += 1

        return U, V, W

    def compute_tensor_fast(U, V, W):
        U_3d = U.reshape(3, 3, -1)
        V_3d = V.reshape(3, 2, -1)
        W_3d = W.reshape(2, 3, -1)
        return np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)

    def check_rounded_solution(U_opt, V_opt, W_opt):
        """Try rounding to various scales and validate exactly."""
        # Try integer, half-integer, quarter-integer, etc.
        for scale in [1, 2, 4, 8, 16, 32]:
            U_r = np.round(U_opt * scale) / scale
            V_r = np.round(V_opt * scale) / scale
            W_r = np.round(W_opt * scale) / scale
            T_check = compute_tensor_fast(U_r, V_r, W_r)
            if np.max(np.abs(T_check - 1.0)) < 1e-12:
                return U_r, V_r, W_r
        return None

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2 using multiple strategies."""
        U17, V17, W17 = get_17_rank_block()

        def residuals(x):
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            T = compute_tensor_fast(U, V, W)
            return (T - 1.0).flatten()

        def objective(x):
            r = residuals(x)
            return np.dot(r, r)

        def polish(x0, max_iter=300):
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter,
                                       ftol=1e-15, xtol=1e-15, gtol=1e-15)
                return result.x, np.max(np.abs(result.fun))
            except Exception:
                return x0, np.max(np.abs(residuals(x0)))

        def polish_bfgs(x0, max_iter=200):
            try:
                from scipy.optimize import minimize
                result = minimize(objective, x0, method='BFGS', options={'maxiter': max_iter, 'gtol': 1e-12})
                return result.x, np.max(np.abs(residuals(result.x)))
            except Exception:
                return x0, np.max(np.abs(residuals(x0)))

        best_result = None
        best_residual = np.inf
        rng = np.random.default_rng(42)

        # Phase 1: Try dropping each of the 17 columns with LM
        for drop_idx in range(17):
            if time_left() < 5:
                break

            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].flatten()
            V_init = V17[:, keep_cols].flatten()
            W_init = W17[:, keep_cols].flatten()
            x0 = np.concatenate([U_init, V_init, W_init])

            try:
                x_opt, residual = polish(x0, max_iter=200)
                
                # If LM gets close, try BFGS polish
                if residual < 1e-3:
                    x_opt2, residual2 = polish_bfgs(x_opt, max_iter=100)
                    if residual2 < residual:
                        x_opt, residual = x_opt2, residual2

                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt

                    if residual < 1e-8:
                        U_opt = x_opt[:144].reshape(9, 16)
                        V_opt = x_opt[144:240].reshape(6, 16)
                        W_opt = x_opt[240:336].reshape(6, 16)
                        rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded
            except Exception:
                pass

        # Phase 2: Random perturbations (increased attempts, lower max_iter)
        for attempt in range(150):
            if time_left() < 5:
                break

            drop_idx = rng.integers(0, 17)
            keep_cols = [c for c in range(17) if c != drop_idx]
            U_init = U17[:, keep_cols].copy()
            V_init = V17[:, keep_cols].copy()
            W_init = W17[:, keep_cols].copy()

            noise_scale = 0.005 + 0.005 * (attempt % 15)
            U_init += rng.normal(0, noise_scale, U_init.shape)
            V_init += rng.normal(0, noise_scale, V_init.shape)
            W_init += rng.normal(0, noise_scale, W_init.shape)

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                x_opt, residual = polish(x0, max_iter=150)
                
                if residual < 1e-3:
                    x_opt2, residual2 = polish_bfgs(x_opt, max_iter=50)
                    if residual2 < residual:
                        x_opt, residual = x_opt2, residual2

                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt

                    if residual < 1e-8:
                        U_opt = x_opt[:144].reshape(9, 16)
                        V_opt = x_opt[144:240].reshape(6, 16)
                        W_opt = x_opt[240:336].reshape(6, 16)
                        rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                        if rounded is not None:
                            return rounded
            except Exception:
                pass

        # Phase 3: Column Merging Strategy
        for i in range(17):
            for j in range(i + 1, 17):
                if time_left() < 5:
                    break

                keep_cols = [c for c in range(17) if c != i and c != j]
                U_new_col = (U17[:, i] + U17[:, j]) / 2.0
                V_new_col = (V17[:, i] + V17[:, j]) / 2.0
                W_new_col = (W17[:, i] + W17[:, j]) / 2.0

                U_init = np.column_stack([U17[:, keep_cols], U_new_col])
                V_init = np.column_stack([V17[:, keep_cols], V_new_col])
                W_init = np.column_stack([W17[:, keep_cols], W_new_col])

                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

                try:
                    x_opt, residual = polish(x0, max_iter=150)
                    
                    if residual < 1e-3:
                        x_opt2, residual2 = polish_bfgs(x_opt, max_iter=50)
                        if residual2 < residual:
                            x_opt, residual = x_opt2, residual2

                    if residual < best_residual:
                        best_residual = residual
                        best_result = x_opt

                        if residual < 1e-8:
                            U_opt = x_opt[:144].reshape(9, 16)
                            V_opt = x_opt[144:240].reshape(6, 16)
                            W_opt = x_opt[240:336].reshape(6, 16)
                            rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                            if rounded is not None:
                                return rounded
                except Exception:
                    pass

        # Phase 4: Final Polish
        if best_result is not None and best_residual < 1e-4:
            if time_left() > 10:
                x_polished, residual_polished = polish(best_result, max_iter=500)
                if residual_polished < best_residual:
                    best_result = x_polished
                    best_residual = residual_polished

                U_opt = best_result[:144].reshape(9, 16)
                V_opt = best_result[144:240].reshape(6, 16)
                W_opt = best_result[240:336].reshape(6, 16)
                rounded = check_rounded_solution(U_opt, V_opt, W_opt)
                if rounded is not None:
                    return rounded

        return None

    def try_find_rank_15_block():
        """Try to find a rank 15 decomposition for 3x3x2 (stretch goal)."""
        U17, V17, W17 = get_17_rank_block()

        def residuals15(x):
            U = x[:135].reshape(9, 15)
            V = x[135:225].reshape(6, 15)
            W = x[225:315].reshape(6, 15)
            T = compute_tensor_fast(U, V, W)
            return (T - 1.0).flatten()

        def polish15(x0, max_iter=400):
            try:
                result = least_squares(residuals15, x0, method='lm', max_nfev=max_iter,
                                       ftol=1e-15, xtol=1e-15, gtol=1e-15)
                return result.x, np.max(np.abs(result.fun))
            except Exception:
                return x0, np.max(np.abs(residuals15(x0)))

        rng = np.random.default_rng(123)
        best_result = None
        best_residual = np.inf

        # Try dropping 2 columns from the 17-rank solution
        for drop1 in range(17):
            for drop2 in range(drop1 + 1, 17):
                if time_left() < 5:
                    return None

                keep_cols = [c for c in range(17) if c != drop1 and c != drop2]
                U_init = U17[:, keep_cols].flatten()
                V_init = V17[:, keep_cols].flatten()
                W_init = W17[:, keep_cols].flatten()
                x0 = np.concatenate([U_init, V_init, W_init])

                try:
                    x_opt, residual = polish15(x0, max_iter=300)

                    if residual < best_residual:
                        best_residual = residual
                        best_result = x_opt

                        if residual < 1e-8:
                            U_opt = x_opt[:135].reshape(9, 15)
                            V_opt = x_opt[135:225].reshape(6, 15)
                            W_opt = x_opt[225:315].reshape(6, 15)
                            for scale in [1, 2, 4, 8]:
                                U_r = np.round(U_opt * scale) / scale
                                V_r = np.round(V_opt * scale) / scale
                                W_r = np.round(W_opt * scale) / scale
                                T_check = compute_tensor_fast(U_r, V_r, W_r)
                                if np.max(np.abs(T_check - 1.0)) < 1e-12:
                                    return U_r, V_r, W_r
                except Exception:
                    pass

        # Random perturbations
        for attempt in range(40):
            if time_left() < 5:
                return None

            drops = rng.choice(17, size=2, replace=False)
            keep_cols = [c for c in range(17) if c not in drops]
            U_init = U17[:, keep_cols].copy()
            V_init = V17[:, keep_cols].copy()
            W_init = W17[:, keep_cols].copy()

            noise_scale = 0.01 + 0.005 * (attempt % 10)
            U_init += rng.normal(0, noise_scale, U_init.shape)
            V_init += rng.normal(0, noise_scale, V_init.shape)
            W_init += rng.normal(0, noise_scale, W_init.shape)

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                x_opt, residual = polish15(x0, max_iter=300)

                if residual < best_residual:
                    best_residual = residual
                    best_result = x_opt

                    if residual < 1e-8:
                        U_opt = x_opt[:135].reshape(9, 15)
                        V_opt = x_opt[135:225].reshape(6, 15)
                        W_opt = x_opt[225:315].reshape(6, 15)
                        for scale in [1, 2, 4, 8]:
                            U_r = np.round(U_opt * scale) / scale
                            V_r = np.round(V_opt * scale) / scale
                            W_r = np.round(W_opt * scale) / scale
                            T_check = compute_tensor_fast(U_r, V_r, W_r)
                            if np.max(np.abs(T_check - 1.0)) < 1e-12:
                                return U_r, V_r, W_r
            except Exception:
                pass

        if best_result is not None and best_residual < 1e-4 and time_left() > 10:
            x_polished, residual_polished = polish15(best_result, max_iter=800)
            if residual_polished < best_residual:
                U_opt = x_polished[:135].reshape(9, 15)
                V_opt = x_polished[135:225].reshape(6, 15)
                W_opt = x_polished[225:315].reshape(6, 15)
                for scale in [1, 2, 4, 8]:
                    U_r = np.round(U_opt * scale) / scale
                    V_r = np.round(V_opt * scale) / scale
                    W_r = np.round(W_opt * scale) / scale
                    T_check = compute_tensor_fast(U_r, V_r, W_r)
                    if np.max(np.abs(T_check - 1.0)) < 1e-12:
                        return U_r, V_r, W_r

        return None

    # Try to find rank 16 block (with time limit)
    block_16 = None
    if time_left() > 15:
        block_16 = try_find_rank_16_block()

    # Try rank 15 if time permits
    block_15 = None
    if block_16 is None and time_left() > 10:
        block_15 = try_find_rank_15_block()

    if block_15 is not None:
        U_block, V_block, W_block = block_15
        rank_block = 15
    elif block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17

    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total))  # 3*8
    W = np.zeros((24, rank_total))  # 8*3

    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block

        # U
        U[:, col_start:col_end] = U_block

        # V
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j * 8 + k_global, col_start + c] = V_block[j * 2 + k_local, c]

        # W
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global * 3 + i, col_start + c] = W_block[k_local * 3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if validate(U, V, W):
        return result
    else:
        return schoolbook()
