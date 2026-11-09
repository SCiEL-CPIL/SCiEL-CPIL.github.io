import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 iff (i,j,k) consistent where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    T_flat = T.flatten()

    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        return np.einsum('ar,br,cr->abc', U, V, W)

    def objective_from_params(params, rank):
        """Compute sum of squared errors from flat parameter vector."""
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        U = params[:U_size].reshape((n * m, rank))
        V = params[U_size:U_size + V_size].reshape((m * p, rank))
        W = params[U_size + V_size:].reshape((p * n, rank))
        T_rec = compute_tensor(U, V, W)
        err = T_rec.flatten() - T_flat
        return np.dot(err, err)

    def check_half_integer(X, tol=1e-8):
        """Check if all entries of X are half-integers (within tolerance)."""
        X2 = 2.0 * X
        return np.allclose(X2, np.round(X2), atol=tol)

    def snap_half_integer(X):
        """Snap entries to nearest half-integer."""
        return 0.5 * np.round(2.0 * X)

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose T and have half-integer entries."""
        T_rec = compute_tensor(U, V, W)
        if not np.allclose(T_rec, T, atol=tol):
            return False
        if not check_half_integer(U, tol=1e-6):
            return False
        if not check_half_integer(V, tol=1e-6):
            return False
        if not check_half_integer(W, tol=1e-6):
            return False
        return True

    def polish(params, rank, max_iter=2000, ftol=1e-14):
        """High-precision polish using L-BFGS-B with tight tolerances."""
        from scipy.optimize import minimize
        try:
            res = minimize(objective_from_params, params, args=(rank,),
                           method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-12,
                                    'maxfun': max_iter * 3})
            return res.x, res.fun
        except Exception:
            return params, objective_from_params(params, rank)

    def extract_factors(params, rank):
        U_size = n * m * rank
        V_size = m * p * rank
        U = params[:U_size].reshape((n * m, rank))
        V = params[U_size:U_size + V_size].reshape((m * p, rank))
        W = params[U_size + V_size:].reshape((p * n, rank))
        return U, V, W

    start_time = time.time()
    time_limit = 130.0  # Leave 20s margin before 150s kill

    # Strategy: Try ranks from 17 to 23
    # For each rank, use multiple random initializations with different strategies
    # Then polish the best candidates and check for half-integer solutions

    best_result = None
    best_rank = rank_schoolbook

    for rank in range(17, 24):
        if time.time() - start_time > time_limit:
            break

        n_params = (n * m + m * p + p * n) * rank
        best_obj = np.inf
        best_params = None

        # Multiple random restarts with different initialization strategies
        n_restarts = 15 if rank <= 19 else 10
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break

            # Different initialization strategies
            if restart == 0:
                # Small random
                np.random.seed(1000 + rank * 100 + restart)
                scale = 0.1
            elif restart == 1:
                # Medium random
                np.random.seed(2000 + rank * 100 + restart)
                scale = 0.5
            elif restart == 2:
                # Larger random
                np.random.seed(3000 + rank * 100 + restart)
                scale = 1.0
            elif restart == 3:
                # Sparse-like: mostly zeros with some ones
                np.random.seed(4000 + rank * 100 + restart)
                U0 = np.zeros((n * m, rank))
                V0 = np.zeros((m * p, rank))
                W0 = np.zeros((p * n, rank))
                for r in range(rank):
                    # Randomly pick which entries are nonzero
                    for idx in np.random.choice(n * m, size=2, replace=False):
                        U0[idx, r] = np.random.choice([-1, 1])
                    for idx in np.random.choice(m * p, size=2, replace=False):
                        V0[idx, r] = np.random.choice([-1, 1])
                    for idx in np.random.choice(p * n, size=2, replace=False):
                        W0[idx, r] = np.random.choice([-1, 1])
                params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                # Quick objective check
                obj = objective_from_params(params0, rank)
                if obj < best_obj:
                    best_obj = obj
                    best_params = params0.copy()
                continue
            else:
                # Various random scales
                np.random.seed(5000 + rank * 100 + restart * 7)
                scale = 0.1 * (1 + 0.3 * restart)

            U0 = np.random.randn(n * m, rank) * scale
            V0 = np.random.randn(m * p, rank) * scale
            W0 = np.random.randn(p * n, rank) * scale
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            # Use L-BFGS-B for optimization
            from scipy.optimize import minimize
            try:
                res = minimize(objective_from_params, params0, args=(rank,),
                               method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                obj = res.fun
                if obj < best_obj:
                    best_obj = obj
                    best_params = res.x.copy()
            except Exception:
                pass

        # If we found a good solution, try to polish and snap
        if best_params is not None and best_obj < 1e-6:
            # Polish
            polished_params, polished_obj = polish(best_params, rank, max_iter=3000, ftol=1e-15)
            if polished_obj < 1e-10:
                U, V, W = extract_factors(polished_params, rank)
                # Try snapping
                U_snap = snap_half_integer(U)
                V_snap = snap_half_integer(V)
                W_snap = snap_half_integer(W)
                if verify_decomposition(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                # Also try without snapping (in case it's already exact)
                if verify_decomposition(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 24
    rank = rank_schoolbook
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
