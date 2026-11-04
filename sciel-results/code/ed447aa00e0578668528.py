import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _generate_target_tensor(n, m, p):
    """Generate the target tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _compute_residuals(params, n, m, p, rank, T_flat):
    """Compute residuals for a given parameter vector."""
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    
    # T_hat[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    residuals = T_hat - T_flat.reshape(n * m, m * p, p * n)
    return residuals.flatten()

def _try_rank(n, m, p, target_rank, T_flat, timeout=2.0):
    """Try to find a decomposition for the given target rank."""
    start_time = time.time()
    
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_params = size_u + size_v + size_w
    
    best_resid = np.inf
    best_params = None
    
    # Try a few random initializations
    # Increase trials if we have time, but keep it bounded
    for trial in range(5):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 999 + 123)
        # Initialize with small random values
        # Scale initialization to help convergence
        x0 = np.random.randn(total_params) * 0.5
        
        def objective(params):
            return _compute_residuals(params, n, m, p, target_rank, T_flat)
        
        try:
            # Use least_squares which is robust for nonlinear least squares
            result = least_squares(objective, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            final_resid = np.sum(result.fun * result.fun)
            if final_resid < best_resid:
                best_resid = final_resid
                best_params = result.x
        except Exception:
            continue
    
    if best_params is not None and best_resid < 1e-8:
        U = best_params[:size_u].reshape(n * m, target_rank)
        V = best_params[size_u:size_u + size_v].reshape(m * p, target_rank)
        W = best_params[size_u + size_v:].reshape(p * n, target_rank)
        return U, V, W
        
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Keep 10s margin for 150s limit
    
    T = _generate_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 21
    # We try from 41 down to 21
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    for target_rank in range(41, 20, -1):
        if time.time() > deadline:
            break
            
        # Allocate less time for higher ranks (closer to 42) as they are less likely to be minimal if 42 is minimal?
        # Actually, if 42 is minimal, 41 will fail.
        # If 41 is minimal, it might be easier to find?
        # Let's give 2 seconds per rank.
        result = _try_rank(n, m, p, target_rank, T_flat, timeout=2.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
