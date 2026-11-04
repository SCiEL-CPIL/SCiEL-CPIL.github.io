import numpy as np
import time
from scipy.optimize import minimize

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
    # Unpack params
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    
    # Compute T_hat
    # T_hat[idx_u, idx_v, idx_w] = sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
    # This is a tensor product sum.
    # We can compute this as:
    # For each r, T_hat += np.einsum('ir,jr,kr->ijk', U, V, W)
    # But this is slow.
    # Alternative: T_hat = U @ V.T @ W? No.
    # T_hat[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    
    # Vectorized approach:
    # Reshape T_hat to (n*m, m*p, p*n)
    # We can compute the sum over r using broadcasting?
    # U[:, :, None] * V[None, :, :] * W[None, None, :] -> (nm, mp, np, rank)
    # Sum over rank axis.
    # Memory: 6 * 21 * 14 * rank. If rank=42, 6*21*14*42 ~ 75k floats. Small.
    
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Residuals
    residuals = T_hat - T_flat.reshape(n * m, m * p, p * n)
    return residuals.flatten()

def _try_rank(n, m, p, target_rank, T_flat, timeout=5.0):
    """Try to find a decomposition for the given target rank."""
    start_time = time.time()
    
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_params = size_u + size_v + size_w
    
    best_resid = np.inf
    best_params = None
    
    # Try a few random initializations
    for trial in range(10):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(trial * 12345 + 42)
        # Initialize with small random values
        x0 = np.random.randn(total_params) * 0.1
        
        # Define objective
        def objective(params):
            res = _compute_residuals(params, n, m, p, target_rank, T_flat)
            return np.sum(res * res)
        
        try:
            # Use L-BFGS-B
            result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 200})
            final_resid = result.fun
            if final_resid < best_resid:
                best_resid = final_resid
                best_params = result.x
        except Exception:
            continue
    
    if best_params is not None and best_resid < 1e-6:
        # Extract U, V, W
        U = best_params[:size_u].reshape(n * m, target_rank)
        V = best_params[size_u:size_u + size_v].reshape(m * p, target_rank)
        W = best_params[size_u + size_v:].reshape(p * n, target_rank)
        return U, V, W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    T = _generate_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    
    # Search for lower rank
    # Lower bound is max(n*m, m*p, p*n) = 21
    for target_rank in range(21, 42):
        if time.time() > deadline:
            break
            
        result = _try_rank(n, m, p, target_rank, T_flat, timeout=10.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

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
