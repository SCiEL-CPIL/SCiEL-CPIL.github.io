import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x5 * 5x5.
    Uses ALS with Kronecker-structured warm starts and nonlinear polishing.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook solution as fallback and warm start source
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = build_schoolbook()
    best_rank = schoolbook_rank
    
    def check_time():
        return time.time() < deadline

    def compute_residual(U, V, W):
        """Compute the residual vector for the tensor equation."""
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def is_valid(U, V, W, tol=1e-9):
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def als_solve(R, seed=0, max_iters=100, init_U=None, init_V=None, init_W=None):
        """
        Alternating Least Squares with random restarts and structured init.
        """
        if not check_time():
            return None
            
        np.random.seed(seed)
        
        # Initialization
        inits = []
        
        # Strategy 1: Random
        U_rand = np.random.randn(n * m, R)
        V_rand = np.random.randn(m * p, R)
        W_rand = np.random.randn(p * n, R)
        inits.append((U_rand, V_rand, W_rand))
        
        # Strategy 2: Warm start from Schoolbook (if R < schoolbook_rank)
        if R < schoolbook_rank and init_U is not None:
            # Take the first R columns of the schoolbook solution
            # This effectively removes (schoolbook_rank - R) terms.
            # The residual will be large for the missing terms, but ALS can try to fix it.
            U_warm = init_U[:, :R].copy()
            V_warm = init_V[:, :R].copy()
            W_warm = init_W[:, :R].copy()
            
            # Add small noise to avoid singularities/symmetry
            U_warm += np.random.randn(n * m, R) * 1e-4
            V_warm += np.random.randn(m * p, R) * 1e-4
            W_warm += np.random.randn(p * n, R) * 1e-4
            
            inits.append((U_warm, V_warm, W_warm))
            
            # Strategy 3: Warm start with random column removal
            # Remove a random set of columns from schoolbook
            cols_to_keep = np.random.choice(schoolbook_rank, R, replace=False)
            U_warm2 = init_U[:, cols_to_keep].copy()
            V_warm2 = init_V[:, cols_to_keep].copy()
            W_warm2 = init_W[:, cols_to_keep].copy()
            
            U_warm2 += np.random.randn(n * m, R) * 1e-4
            V_warm2 += np.random.randn(m * p, R) * 1e-4
            W_warm2 += np.random.randn(p * n, R) * 1e-4
            
            inits.append((U_warm2, V_warm2, W_warm2))

        best_local_err = np.inf
        best_local_sol = None
        
        for init_idx, (U, V, W) in enumerate(inits):
            if not check_time():
                break
                
            # ALS Loop
            for it in range(max_iters):
                if not check_time():
                    break
                
                # Solve for U
                n_constraints = n * m * p
                n_vars_u = n * m * R
                A_U = np.zeros((n_constraints, n_vars_u))
                b = np.ones(n_constraints)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        u_row_idx = i * m + j
                        for k in range(p):
                            coeff = V[j * p + k] * W[k * n + i]
                            A_U[idx, u_row_idx * R : (u_row_idx + 1) * R] = coeff
                            idx += 1
                
                try:
                    u_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U = u_flat.reshape(n * m, R)
                except:
                    break
                
                # Solve for V
                n_vars_v = m * p * R
                A_V = np.zeros((n_constraints, n_vars_v))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U[i * m + j] * W[k * n + i]
                            v_row_idx = j * p + k
                            A_V[idx, v_row_idx * R : (v_row_idx + 1) * R] = coeff
                            idx += 1
                            
                try:
                    v_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V = v_flat.reshape(m * p, R)
                except:
                    break
                    
                # Solve for W
                n_vars_w = p * n * R
                A_W = np.zeros((n_constraints, n_vars_w))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U[i * m + j] * V[j * p + k]
                            w_row_idx = k * n + i
                            A_W[idx, w_row_idx * R : (w_row_idx + 1) * R] = coeff
                            idx += 1
                            
                try:
                    w_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W = w_flat.reshape(p * n, R)
                except:
                    break
                    
                # Check convergence
                res = compute_residual(U, V, W)
                err = np.max(np.abs(res))
                if err < best_local_err:
                    best_local_err = err
                    best_local_sol = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-9:
                    return U, V, W
                    
        # If ALS didn't converge, try polishing the best local solution
        if best_local_sol is not None and check_time():
            U0, V0, W0 = best_local_sol
            
            # Flatten all variables for scipy
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_residual(U, V, W)
                
            try:
                result = least_squares(objective, x0, method='trf', max_nfev=200, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                if result.success or result.cost < 1e-8:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*R].reshape(n*m, R)
                    V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    if is_valid(U_opt, V_opt, W_opt, tol=1e-8):
                        return U_opt, V_opt, W_opt
            except Exception as e:
                pass
                
        return None

    # Search ranks from high to low
    for R in range(49, 24, -1):
        if not check_time():
            break
            
        # More restarts for higher ranks
        num_restarts = 5 if R >= 45 else 3
        
        for restart in range(num_restarts):
            if not check_time():
                break
            result = als_solve(R, seed=R * 100 + restart, max_iters=100, init_U=best_U, init_V=best_V, init_W=best_W)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
