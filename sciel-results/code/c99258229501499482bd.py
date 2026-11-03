"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try ranks from schoolbook-1 down to 13
    for rank in range(schoolbook_rank - 1, 12, -1):
        if time.time() > deadline:
            break

        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue

    return best_result

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(S - 1.0))
    return max_err < 1e-9

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using least_squares."""
    from scipy.optimize import least_squares

    # Residual function for least_squares
    def residual_function(params):
        U = params[:n * m * rank].reshape(n, m, rank)
        V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
        W = params[n * m * rank + m * p * rank:].reshape(p, n, rank)

        # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)

        residuals = S.flatten() - 1.0
        return residuals

    # Multiple restarts to escape local minima
    np.random.seed(rank * 1000)

    for restart in range(30):
        if time.time() > deadline:
            break

        # Random initialization with different strategies
        if restart % 3 == 0:
            # Standard random
            U0 = np.random.randn(n * m, rank)
            V0 = np.random.randn(m * p, rank)
            W0 = np.random.randn(p * n, rank)
            scale = 1.0 / np.sqrt(rank)
            U0 *= scale
            V0 *= scale
            W0 *= scale
        elif restart % 3 == 1:
            # Sparse initialization
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            for _ in range(rank):
                idx_u = np.random.randint(0, n * m)
                idx_v = np.random.randint(0, m * p)
                idx_w = np.random.randint(0, p * n)
                U0[idx_u, _] = np.random.choice([-1, 1])
                V0[idx_v, _] = np.random.choice([-1, 1])
                W0[idx_w, _] = np.random.choice([-1, 1])
        else:
            # Structured initialization based on schoolbook
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < rank:
                            U0[i * m + j, col] = 1
                            V0[j * p + k, col] = 1
                            W0[k * n + i, col] = 1
                            col += 1
            # Add random perturbation
            U0 += 0.1 * np.random.randn(n * m, rank)
            V0 += 0.1 * np.random.randn(m * p, rank)
            W0 += 0.1 * np.random.randn(p * n, rank)

        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            result = least_squares(residual_function, params0, method='trf',
                                   max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
        except Exception:
            continue

        # Check if cost is small enough
        if result.cost < 1e-10:
            U_opt = result.x[:n * m * rank].reshape(n, m, rank)
            V_opt = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
            W_opt = result.x[n * m * rank + m * p * rank:].reshape(p, n, rank)

            # Try snapping to half-integers and integers
            for denom in [1, 2, 3, 4, 5, 6]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom

                # Verify exactness
                if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }

            # If no exact snap found, try to use the optimized values directly
            # but only if they are very close to half-integers
            U_close = np.allclose(U_opt, np.round(U_opt * 2) / 2, atol=1e-6)
            V_close = np.allclose(V_opt, np.round(V_opt * 2) / 2, atol=1e-6)
            W_close = np.allclose(W_opt, np.round(W_opt * 2) / 2, atol=1e-6)
            
            if U_close and V_close and W_close:
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                if _verify_exact(U_snap, V_snap, W_snap, n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }

    return None
