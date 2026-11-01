import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _pairwise_sq_dists(X):
    """Compute squared pairwise distances."""
    # X is (N, D)
    # dists[i, j] = ||X[i] - X[j]||^2
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2)
    return dists

def _objective_and_grad(X_flat, beta=10.0, lambda_scale=1.0):
    """
    Objective: smooth ratio of max to min squared distance.
    Plus a penalty to fix scale (variance = 1).
    Returns (value, gradient).
    """
    X = X_flat.reshape(N, D)
    
    # Center X
    mean_X = np.mean(X, axis=0)
    X_c = X - mean_X
    
    # Compute squared distances
    diff = X_c[:, np.newaxis, :] - X_c[np.newaxis, :, :]
    dists = np.sum(diff**2, axis=2)
    
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_flat = dists[mask]
    
    # Log-sum-exp for max and min
    # Max: (1/beta) * log(sum(exp(beta * d)))
    # Min: -(1/beta) * log(sum(exp(-beta * d)))
    
    # Numerical stability
    max_d = np.max(d_flat)
    min_d = np.min(d_flat)
    
    # For max
    exp_max = np.exp(beta * (d_flat - max_d))
    sum_exp_max = np.sum(exp_max)
    log_sum_max = max_d + (1/beta) * np.log(sum_exp_max)
    
    # For min
    exp_min = np.exp(-beta * (d_flat - min_d))
    sum_exp_min = np.sum(exp_min)
    log_sum_min = -min_d - (1/beta) * np.log(sum_exp_min)
    
    # Objective value
    val = log_sum_max / (-log_sum_min)
    
    # Scale penalty
    var = np.mean(np.sum(X_c**2, axis=1))
    penalty = (var - 1.0)**2
    val += lambda_scale * penalty
    
    # Gradient
    # d(val)/d(d_ij)
    # val = U / V where U = log_sum_max, V = -log_sum_min
    # dU/dd_ij = (beta * exp(beta*(d_ij - max_d))) / sum_exp_max
    # dV/dd_ij = - ( -beta * exp(-beta*(d_ij - min_d))) / sum_exp_min = (beta * exp(-beta*(d_ij - min_d))) / sum_exp_min
    
    dU_dd = beta * exp_max / sum_exp_max
    dV_dd = beta * exp_min / sum_exp_min
    
    dval_dd = (dU_dd * V - U * dV_dd) / (V**2)
    
    # Map dval_dd back to matrix
    dval_dd_mat = np.zeros((N, N))
    dval_dd_mat[mask] = dval_dd
    dval_dd_mat = dval_dd_mat + dval_dd_mat.T # Symmetrize
    
    # Gradient w.r.t X
    # d(d_ij)/d(X_k) = 2 * (X_k - X_j) if i=k, j!=k
    #                 = -2 * (X_k - X_i) if j=k, i!=k
    # Sum over j for fixed k:
    # grad_k = sum_j dval_dd_mat[k,j] * 2 * (X_k - X_j)
    
    # Vectorized:
    # grad_X = 2 * sum_j dval_dd_mat[k,j] * (X_k - X_j)
    # Note: dval_dd_mat[k,k] = 0, so it doesn't matter.
    
    # grad_X[k] = 2 * ( sum_j dval_dd_mat[k,j] X_k - sum_j dval_dd_mat[k,j] X_j )
    #           = 2 * ( X_k * sum_j dval_dd_mat[k,j] - (dval_dd_mat @ X)[k] )
    
    row_sums = np.sum(dval_dd_mat, axis=1)
    term1 = X * row_sums[:, np.newaxis]
    term2 = dval_dd_mat @ X
    grad_obj = 2.0 * (term1 - term2)
    
    # Gradient of penalty
    # P = (var - 1)^2
    # dP/dX_k = 2(var-1) * 2/N * (X_k - mean_X)
    # Note: X_c = X - mean_X
    grad_pen = lambda_scale * 4.0 * (var - 1.0) / N * X_c
    
    grad_total = grad_obj + grad_pen
    
    return val, grad_total.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial random points
    # Try to start with a good structure, e.g. vertices of a polytope or random
    
    # We will do multiple restarts
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale to unit variance
        X_init = np.random.randn(N, D)
        X_init -= np.mean(X_init, axis=0)
        var = np.mean(np.sum(X_init**2, axis=1))
        X_init /= np.sqrt(var)
        
        # Optimize
        try:
            res = minimize(
                _objective_and_grad, 
                X_init.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
            
        if not res.success:
            # Still might be better than current best
            pass
            
        X_opt = res.x.reshape(N, D)
        
        # Compute true score
        # Center and scale check? The penalty should have kept it near unit variance.
        # But we should compute the raw ratio on the returned points.
        # The raw score is (max_dist / min_dist)^2.
        
        # Ensure distinct points
        if np.any(np.isclose(X_opt[:, np.newaxis, :], X_opt[np.newaxis, :, :], atol=1e-8)):
             # If points are too close, it's bad. 
             # The min distance will be ~0, ratio inf.
             continue

        dists = _pairwise_sq_dists(X_opt)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_flat = dists[mask]
        
        min_d = np.min(d_flat)
        max_d = np.max(d_flat)
        
        if min_d < 1e-12:
            score = float('inf')
        else:
            score = (max_d / min_d)**2
            
        if score < best_score:
            best_score = score
            best_points = X_opt.copy()
            
    if best_points is None:
        # Fallback
        return np.random.randn(N, D)
        
    return best_points
# EVOLVE-BLOCK-END
