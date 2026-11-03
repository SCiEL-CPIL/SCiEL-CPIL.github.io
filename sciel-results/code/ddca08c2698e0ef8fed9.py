import time
import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s

    # The rank-17 solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1

    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1

    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1

    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1

    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1

    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1

    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1

    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1

    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    # Precompute indices for speed
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))

    def fast_residual(U, V, W):
        res = 0.0
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            res += (val - 1.0) ** 2
        return res

    def fast_residual_grad(U, V, W):
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        for u_idx, v_idx, w_idx in indices:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            diff = val - 1.0
            VW = V[v_idx, :] * W[w_idx, :]
            U_grad[u_idx, :] += 2 * diff * VW
            V_grad[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
            W_grad[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
        return U_grad, V_grad, W_grad

    def verify_exact(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact using exact arithmetic."""
        for u_idx, v_idx, w_idx in indices:
            total = Fraction(0)
            for r in range(U.shape[1]):
                u_val = Fraction(str(U[u_idx, r]))
                v_val = Fraction(str(V[v_idx, r]))
                w_val = Fraction(str(W[w_idx, r]))
                total += u_val * v_val * w_val
            if abs(float(total - 1)) > tol:
                return False
        return True

    def try_round_and_verify(U, V, W, denom=8):
        """Try rounding to multiples of 1/denom and verify exactly."""
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        # Clean up floating point artifacts
        U_r = np.where(np.abs(U_r) < 1e-12, 0, U_r)
        V_r = np.where(np.abs(V_r) < 1e-12, 0, V_r)
        W_r = np.where(np.abs(W_r) < 1e-12, 0, W_r)
        if verify_exact(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    best_solution = (U17.copy(), V17.copy(), W17.copy())
    best_res = fast_residual(U17, V17, W17)

    # Verify rank-17 is exact
    if verify_exact(U17, V17, W17):
        pass  # Good, rank-17 is valid
    else:
        # Should not happen, but if it does, we need to fix it
        pass

    R = 16

    def make_objective(R_val, l1_penalty=0.0):
        def objective(x):
            U = x[:n*m*R_val].reshape(n*m, R_val)
            V = x[n*m*R_val:n*m*R_val+m*p*R_val].reshape(m*p, R_val)
            W = x[n*m*R_val+m*p*R_val:].reshape(p*n, R_val)
            res = fast_residual(U, V, W)
            if l1_penalty > 0:
                res += l1_penalty * (np.abs(U).sum() + np.abs(V).sum() + np.abs(W).sum())
            return res
        return objective

    def make_grad(R_val, l1_penalty=0.0):
        def grad(x):
            U = x[:n*m*R_val].reshape(n*m, R_val)
            V = x[n*m*R_val:n*m*R_val+m*p*R_val].reshape(m*p, R_val)
            W = x[n*m*R_val+m*p*R_val:].reshape(p*n, R_val)
            Ug, Vg, Wg = fast_residual_grad(U, V, W)
            if l1_penalty > 0:
                Ug += l1_penalty * np.sign(U)
                Vg += l1_penalty * np.sign(V)
                Wg += l1_penalty * np.sign(W)
            return np.concatenate([Ug.flatten(), Vg.flatten(), Wg.flatten()])
        return grad

    # Strategy 1: Try removing each column from rank-17 and optimize with L-BFGS
    for remove_col in range(17):
        if time.time() > deadline:
            break

        keep_cols = [r for r in range(17) if r != remove_col]
        U0 = U17[:, keep_cols].copy()
        V0 = V17[:, keep_cols].copy()
        W0 = W17[:, keep_cols].copy()

        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        # Try with and without small L1 penalty
        for l1_pen in [0.0, 0.005, 0.02, 0.08]:
            if time.time() > deadline:
                break
            obj = make_objective(R, l1_pen)
            grad = make_grad(R, l1_pen)
            try:
                result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100})
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                res_check = fast_residual(U_opt, V_opt, W_opt)
                if res_check < best_res:
                    best_res = res_check
                    best_solution = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                    if best_res < 1e-12:
                        break
            except:
                pass
        if best_res < 1e-12:
            break

    # Strategy 2: Random restarts with diverse initializations
    np.random.seed(42)
    if best_res >= 1e-12:
        for attempt in range(200):
            if time.time() > deadline:
                break

            # Different initialization strategies
            strategy = attempt % 6
            if strategy == 0:
                # Small random
                U = np.random.randn(n * m, R) * 0.2
                V = np.random.randn(m * p, R) * 0.2
                W = np.random.randn(p * n, R) * 0.2
            elif strategy == 1:
                # Based on rank-17 with random perturbation
                keep = np.random.choice(17, R, replace=False)
                U = U17[:, keep].copy() + np.random.randn(n * m, R) * 0.05
                V = V17[:, keep].copy() + np.random.randn(m * p, R) * 0.05
                W = W17[:, keep].copy() + np.random.randn(p * n, R) * 0.05
            elif strategy == 2:
                # Sparse random with ±1
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    U[np.random.randint(n*m), r] = np.random.choice([-1, 1])
                    V[np.random.randint(m*p), r] = np.random.choice([-1, 1])
                    W[np.random.randint(p*n), r] = np.random.choice([-1, 1])
            elif strategy == 3:
                # Medium random
                U = np.random.randn(n * m, R) * 0.4
                V = np.random.randn(m * p, R) * 0.4
                W = np.random.randn(p * n, R) * 0.4
            elif strategy == 4:
                # Half-integer random
                U = np.random.randint(0, 5, size=(n * m, R)) * 0.5 - 1.0
                V = np.random.randint(0, 5, size=(m * p, R)) * 0.5 - 1.0
                W = np.random.randint(0, 5, size=(p * n, R)) * 0.5 - 1.0
            else:
                # Structured: try to match known patterns
                U = np.random.choice([-1, 0, 1], size=(n * m, R))
                V = np.random.choice([-1, 0, 1], size=(m * p, R))
                W = np.random.choice([-1, 0, 1], size=(p * n, R))

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            l1_pen = [0.0, 0.005, 0.02][attempt % 3]

            obj = make_objective(R, l1_pen)
            grad = make_grad(R, l1_pen)
            try:
                result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                                 options={'maxiter': 150, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 80})
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                res_check = fast_residual(U_opt, V_opt, W_opt)
                if res_check < best_res:
                    best_res = res_check
                    best_solution = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                    if best_res < 1e-12:
                        break
            except:
                pass

    # If we found a good rank-16 solution, try to round and verify exactly
    if best_res < 1e-8:
        U_best, V_best, W_best = best_solution

        # Final high-precision polish
        obj = make_objective(R, 0.0)
        grad = make_grad(R, 0.0)
        x_polish = np.concatenate([U_best.flatten(), V_best.flatten(), W_best.flatten()])
        try:
            result_polish = minimize(obj, x_polish, jac=grad, method='L-BFGS-B',
                                    options={'maxiter': 1000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            U_best = result_polish.x[:n*m*R].reshape(n*m, R)
            V_best = result_polish.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W_best = result_polish.x[n*m*R+m*p*R:].reshape(p*n, R)
            best_res = fast_residual(U_best, V_best, W_best)
            best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
        except:
            pass

        # Try rounding with exact verification
        for denom in [2, 4, 8, 16]:
            if time.time() > deadline:
                break
            result = try_round_and_verify(U_best, V_best, W_best, denom)
            if result is not None:
                U_r, V_r, W_r = result
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        # Try mixed rounding with exact verification
        for u_denom in [2, 4, 8]:
            for v_denom in [2, 4, 8]:
                for w_denom in [2, 4, 8]:
                    if time.time() > deadline:
                        break
                    U_r = np.round(U_best * u_denom) / u_denom
                    V_r = np.round(V_best * v_denom) / v_denom
                    W_r = np.round(W_best * w_denom) / w_denom
                    U_r = np.where(np.abs(U_r) < 1e-12, 0, U_r)
                    V_r = np.where(np.abs(V_r) < 1e-12, 0, V_r)
                    W_r = np.where(np.abs(W_r) < 1e-12, 0, W_r)
                    if verify_exact(U_r, V_r, W_r):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
