"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def _strassen_rank21():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 2 * b
                elif l == 1: v_idx = 2 * b + 1
                elif l == 2: v_idx = 6 + 2 * b
                elif l == 3: v_idx = 7 + 2 * b
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U0, V0, W0, target_rank, deadline):
    """Try to reduce rank by optimizing from the rank-21 solution."""
    n, m, p = DIMENSIONS
    r0 = U0.shape[1]
    if target_rank >= r0:
        return None
    
    # We'll try to find a rank-target decomposition by solving the system
    # using a nonlinear solver on the factor matrices.
    
    best = None
    best_err = float('inf')
    
    # Try multiple random initializations near the rank-21 solution
    np.random.seed(42)
    max_iters = 200
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        # Initialize with the rank-21 solution, zero out some columns
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Copy first target_rank columns from the rank-21 solution
        U[:, :target_rank] = U0[:, :target_rank]
        V[:, :target_rank] = V0[:, :target_rank]
        W[:, :target_rank] = W0[:, :target_rank]
        
        # Add small random perturbation
        U += 0.01 * np.random.randn(n*m, target_rank)
        V += 0.01 * np.random.randn(m*p, target_rank)
        W += 0.01 * np.random.randn(p*n, target_rank)
        
        # Use scipy's least_squares to minimize the error
        try:
            from scipy.optimize import least_squares
            
            def residuals(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                W_flat = params[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                            res[idx] = s - 1.0
                            idx += 1
                return res
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use bounded optimization to keep values reasonable
            bounds = (np.full_like(x0, -10), np.full_like(x0, 10))
            
            result = least_squares(residuals, x0, bounds=bounds, max_nfev=5000, xtol=1e-10, ftol=1e-10)
            
            if result.success or result.fun[-1] < best_err:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank:(n*m+m*p)*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[(n*m+m*p)*target_rank:].reshape(p*n, target_rank)
                
                err = np.max(np.abs(result.fun))
                if err < best_err:
                    best_err = err
                    best = (U_opt, V_opt, W_opt)
                    
                if err < 1e-8:
                    break
        except Exception:
            continue
    
    if best is not None and best_err < 1e-8:
        U_opt, V_opt, W_opt = best
        # Round to nearest half-integer
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        if _verify(U_round, V_round, W_round, tol=1e-9):
            return (U_round, V_round, W_round)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with the known rank-21 solution
    U21, V21, W21 = _strassen_rank21()
    
    # Try to reduce to rank 20
    result = _try_reduce_rank(U21, V21, W21, 20, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 21
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
# EVOLVE-BLOCK-END
