import numpy as np
import time
from scipy.optimize import least_squares, minimize

_DIMS = (3, 3, 6)
_DEADLINE = 145.0  # safety margin before 150s kill

def _make_residual(n, m, p, R):
    """Return residual(x) for tensor decomposition of T[i,j,k]=1."""
    def residual(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return (T - 1.0).ravel()
    return residual

def _polish(x0, residual, max_nfev=5000):
    """High-precision polish using tight tolerances."""
    try:
        res = least_squares(
            residual, x0, method='trf', x_scale='jac',
            max_nfev=max_nfev, ftol=1e-18, xtol=1e-18, gtol=1e-18
        )
        return res
    except Exception:
        return None

def _check_half_int(U, V, W, n, m, p, R):
    """Verify U,V,W (as half-integers) give exact T[i,j,k]=1."""
    Ui = np.round(U * 2).astype(np.int64)
    Vi = np.round(V * 2).astype(np.int64)
    Wi = np.round(W * 2).astype(np.int64)
    U_t = Ui.reshape(n, m, R)
    V_t = Vi.reshape(m, p, R)
    W_t = Wi.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    return np.all(T == 8)  # 2^3 * 1

def construct():
    t0 = time.time()
    n, m, p = _DIMS
    # Schoolbook rank
    R_school = n * m * p  # 54

    def _elapsed():
        return time.time() - t0

    def _remaining():
        return _DEADLINE - _elapsed()

    def _schoolbook():
        R = R_school
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, R

    # Try ranks from 24 up to schoolbook-1
    # The theoretical lower bound for 3x3x6 is at least 27 (from 3x3x3 lower bound of 23, times 2 / something)
    # But let's be practical: try a range of ranks
    # For 3x3x3, best known is 23. For 3x3x6, a simple Kronecker gives 2*23=46.
    # Schoolbook is 54. Let's try to beat 54.
    
    best_result = None
    best_rank = R_school

    # Try various ranks
    ranks_to_try = list(range(24, 54))
    # Sort by rank (lower first)
    ranks_to_try.sort()

    for R in ranks_to_try:
        if _remaining() < 15.0:  # Need at least 15s for final check and fallback
            break

        n_vars = (n*m + m*p + p*n) * R
        residual = _make_residual(n, m, p, R)

        found = False
        n_restarts = 0
        max_restarts = 80 if R <= 30 else 40

        while not found and n_restarts < max_restarts:
            if _remaining() < 10.0:
                break
            n_restarts += 1

            # Different initialization strategies
            if n_restarts == 1:
                # Structured: start from a scaled-down schoolbook pattern
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                # Distribute schoolbook entries across R columns
                for idx in range(min(n*m*p, R)):
                    i, j, k = idx // (m*p), (idx // p) % m, idx % p
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            elif n_restarts == 2:
                # Random with moderate scale
                scale = 0.5 + 0.5 * np.random.rand()
                x0 = np.random.uniform(-scale, scale, n_vars)
            elif n_restarts == 3:
                # Sparse random
                x0 = np.random.uniform(-1, 1, n_vars)
                mask = np.random.rand(n_vars) < 0.3
                x0[~mask] = 0.0
            elif n_restarts == 4:
                # Near-schoolbook perturbation
                U0, V0, W0, _ = _schoolbook()
                # Take first R columns of schoolbook (if R < 54, truncate)
                R_eff = min(R, 54)
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                U[:, :R_eff] = U0[:, :R_eff]
                V[:, :R_eff] = V0[:, :R_eff]
                W[:, :R_eff] = W0[:, :R_eff]
                # Add small random perturbation
                U += 0.1 * np.random.randn(n*m, R)
                V += 0.1 * np.random.randn(m*p, R)
                W += 0.1 * np.random.randn(p*n, R)
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            else:
                # Various random scales
                scale = 0.3 + 1.5 * np.random.rand()
                x0 = np.random.uniform(-scale, scale, n_vars)
                # Occasionally add structure
                if np.random.rand() < 0.3:
                    # Block-diagonal-like structure
                    block_size = max(1, R // 6)
                    for b in range(6):
                        start = b * block_size
                        end = min(start + block_size, R)
                        if end > start:
                            x0[start:end] += np.random.uniform(-0.5, 0.5, end - start)

            try:
                # First optimization pass
                res = least_squares(
                    residual, x0, method='trf', x_scale='jac',
                    max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10
                )

                # If close enough, polish
                if res.cost < 1e-4:
                    res2 = _polish(res.x, residual, max_nfev=3000)
                    if res2 is not None and res2.cost < res.cost:
                        res = res2

                # Check if we have a good solution
                if res.cost < 1e-8:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:2*n*m*R].reshape(m*p, R)
                    W_sol = res.x[2*n*m*R:].reshape(p*n, R)

                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2.0
                    V_round = np.round(V_sol * 2) / 2.0
                    W_round = np.round(W_sol * 2) / 2.0

                    # Verify exactness
                    if _check_half_int(U_round, V_round, W_round, n, m, p, R):
                        found = True
                        best_rank = R
                        best_result = (U_round, V_round, W_round, R)
                        break
                    else:
                        # Try rounding to nearest integer too
                        U_int = np.round(U_sol).astype(float)
                        V_int = np.round(V_sol).astype(float)
                        W_int = np.round(W_sol).astype(float)
                        Ui = np.round(U_int * 2).astype(np.int64)
                        Vi = np.round(V_int * 2).astype(np.int64)
                        Wi = np.round(W_int * 2).astype(np.int64)
                        U_t = Ui.reshape(n, m, R)
                        V_t = Vi.reshape(m, p, R)
                        W_t = Wi.reshape(p, n, R)
                        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                        if np.all(T == 8):
                            found = True
                            best_rank = R
                            best_result = (U_int, V_int, W_int, R)
                            break

            except Exception:
                pass

        if found:
            break

    # If we found a better rank, use it; otherwise schoolbook
    if best_result is not None and best_rank < R_school:
        U, V, W, R = best_result
        # Final verification
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W, R = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
