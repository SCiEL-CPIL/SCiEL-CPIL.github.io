"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-54 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p, tol=1e-9):
    """Check whether decomp exactly decomposes the multiplication tensor."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    # Construct the tensor T of shape (n, m, p)
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Efficient computation using broadcasting
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We need to compute sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note indices:
    # U index: i, j
    # V index: j, k
    # W index: k, i
    
    # Let's compute the product term by term or use einsum
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    
    # This is a bit tricky with standard broadcasting because of the cyclic indices.
    # Let's just loop over i, j, k for verification as it's only 54 iterations.
    # But we can vectorize the inner sum.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_reshaped[i, j, :]
                v_val = V_reshaped[j, k, :]
                w_val = W_reshaped[k, i, :]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True


def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors of the tensor decomposition."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_reshaped[i, j, :]
                v_val = V_reshaped[j, k, :]
                w_val = W_reshaped[k, i, :]
                s = np.sum(u_val * v_val * w_val)
                error += (s - 1.0) ** 2
    return error


def _try_optimize_rank(target_rank, n, m, p, max_iter=1000, seed=42):
    """
    Attempt to find a valid decomposition of target_rank by optimizing
    the coefficients starting from a schoolbook-like initialization.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W with random values or schoolbook subset
    # We start with the first target_rank columns of schoolbook
    # and then optimize all entries to satisfy the equations.
    
    # A better initialization: Use the schoolbook columns, but randomly permute and select?
    # Or just use the first target_rank columns and set the rest to 0?
    # If we just take the first target_rank columns, the error will be high.
    # We need to adjust the values.
    
    # Let's initialize with random values scaled to 1.
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    # Flatten the parameters
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def objective(params):
        U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank + m*p*target_rank], params[n*m*target_rank + m*p*target_rank:]
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        return _compute_error(U, V, W, n, m, p)
    
    # Use L-BFGS-B or Nelder-Mead
    # Nelder-Mead is robust but slow.
    # Let's try a few restarts.
    
    best_params = None
    best_err = float('inf')
    
    for restart in range(5):
        if restart > 0:
            # Perturb initial params
            params = params + rng.randn(len(params)) * 0.1
        
        res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
    
    if best_err < 1e-6:
        U_flat, V_flat, W_flat = best_params[:n*m*target_rank], best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank], best_params[n*m*target_rank + m*p*target_rank:]
        U = U_flat.reshape(n * m, target_rank)
        V = V_flat.reshape(m * p, target_rank)
        W = W_flat.reshape(p * n, target_rank)
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 by 3x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # First, always have the schoolbook decomposition as fallback
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition
    # We try ranks from 53 down to 40
    
    for target_rank in range(schoolbook_rank - 1, 39, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random restarts for this target rank
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            
            result = _try_optimize_rank(target_rank, n, m, p, max_iter=500, seed=seed)
            if result is not None:
                # Verify one more time
                if _verify(result, n, m, p):
                    # Convert to list of lists with float entries
                    U = [[float(x) for x in row] for row in result["U"]]
                    V = [[float(x) for x in row] for row in result["V"]]
                    W = [[float(x) for x in row] for row in result["W"]]
                    return {"U": U, "V": V, "W": W}
    
    # If no improvement found, return schoolbook
    return best
# EVOLVE-BLOCK-END
