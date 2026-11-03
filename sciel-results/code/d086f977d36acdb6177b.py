import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate optimal spacing. Area per circle ~ pi*r^2. Total area pi*1^2.
    # 300 * pi * r^2 <= pi * 1 => r <= 1/sqrt(300) ~ 0.0577.
    # Actual best is ~0.053. Spacing = 2*r ~ 0.106.
    # Hex lattice spacing s.
    s = 0.11 
    points = []
    # Generate grid points
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.9: # Keep inside
                points.append([x, y])
    
    points = np.array(points)
    # Sort by distance to origin to pick the densest center
    dists = np.linalg.norm(points, axis=1)
    points = points[np.argsort(dists)]
    
    # If we have too many, take first N. If too few, shrink spacing.
    if len(points) < N:
        # Fallback: simple grid
        points = np.random.uniform(-0.8, 0.8, (N, 2))
    else:
        points = points[:N]
        
    # Ensure valid start (no overlaps, inside circle)
    # The hex lattice with s=0.11 should be valid for r=0.055 (dist 0.11)
    # Check if any point is outside unit circle
    r_pts = np.linalg.norm(points, axis=1)
    if np.any(r_pts > 1.0):
        points[r_pts > 1.0] = points[r_pts > 1.0] / (r_pts[r_pts > 1.0, None] + 1e-6) * 0.99

    x0 = points.flatten()
    
    def objective_and_grad(x):
        c = x.reshape(N, 2)
        # Compute pairwise distances
        # d[i,j] = ||c[i] - c[j]||
        # Using broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=-1)
        
        # Boundary distances
        b = 1.0 - np.linalg.norm(c, axis=1)
        
        # Soft min approximation
        # We want to maximize min(min_pair_dist/2, min_boundary_dist)
        # Let r_i = d_ij / 2 for pairs, r_k = b_k for boundaries.
        # We want to minimize - soft_min(r).
        # soft_min(r) = -1/alpha * ln( sum( exp(-alpha * r) ) )
        
        # To avoid underflow/overflow with large alpha, we can shift by min_r
        # But for moderate alpha, direct exp is fine.
        # Let's use alpha = 50.
        alpha = 50.0
        
        # Mask upper triangle for pairs
        # We need to sum exp(-alpha * d_ij / 2) for i < j
        # and exp(-alpha * b_i) for i
        
        # Efficiently compute sum of exps
        # Create upper triangular indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = d[i_idx, j_idx]
        
        # Boundary terms
        b_terms = b
        
        # Combine
        # To prevent numerical issues, find the min of the exponents' arguments
        # args = np.concatenate([-alpha * pair_dists / 2, -alpha * b_terms])
        # This is 45k + 300 = 45300 elements.
        
        # Let's just compute the sum.
        # exp(-50 * 0.05) = exp(-2.5) ~ 0.08
        # exp(-50 * 0.01) = exp(-0.5) ~ 0.6
        # Values are small, sum is manageable.
        
        sum_exp = np.sum(np.exp(-alpha * pair_dists / 2.0)) + np.sum(np.exp(-alpha * b_terms))
        
        if sum_exp < 1e-300:
            return 1e10, np.zeros_like(x)
            
        obj = -np.log(sum_exp) / alpha
        
        # Gradient
        # d obj / d x = -1/(alpha * sum_exp) * d sum_exp / d x
        # d sum_exp / d x_k = sum over active terms
        
        # For pair (i,j): term is exp(-alpha * d_ij / 2)
        # d/dx_k exp(-alpha * d_ij / 2) = exp(-alpha * d_ij / 2) * (-alpha/2) * d d_ij / d x_k
        # d d_ij / d c_i = (c_i - c_j) / d_ij
        # d d_ij / d c_j = (c_j - c_i) / d_ij
        
        # For boundary k: term is exp(-alpha * b_k)
        # b_k = 1 - ||c_k||
        # d b_k / d c_k = - c_k / ||c_k||
        # d/dx_k exp(-alpha * b_k) = exp(-alpha * b_k) * (-alpha) * (- c_k / ||c_k||) = alpha * exp(...) * c_k / ||c_k||
        
        grad = np.zeros_like(c)
        
        # Pair contributions
        # We need to accumulate gradients for each i and j
        # weights = exp(-alpha * pair_dists / 2)
        weights = np.exp(-alpha * pair_dists / 2.0)
        
        # diff_ij = c[i] - c[j]
        # grad_i += weights * (-alpha/2) * diff_ij / d_ij
        # grad_j += weights * (-alpha/2) * (-diff_ij) / d_ij = weights * (alpha/2) * diff_ij / d_ij
        
        # Vectorized accumulation is tricky with scatter.
        # N=300 is small, we can use a loop or sparse matrix trick.
        # Let's use a loop for clarity and correctness, it's only 45k iterations.
        # Actually, 45k iterations in Python is ~0.1s. Acceptable.
        
        # Better: Use np.add.at
        # grad_i_accum = np.zeros((N, 2))
        # For each pair (i, j):
        #   factor = weights[k] * (-alpha/2) / pair_dists[k]
        #   grad_i_accum[i] += factor * diff[i, j]
        #   grad_j_accum[j] -= factor * diff[i, j]  # Note: d/d c_j is opposite
        
        # Let's try to vectorize with matrix operations if possible, or just use the loop.
        # Loop:
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d_ij = pair_dists[k]
            if d_ij < 1e-6:
                continue
            w = weights[k]
            factor = w * (-alpha / 2.0) / d_ij
            diff_ij = c[i] - c[j]
            grad[i] += factor * diff_ij
            grad[j] -= factor * diff_ij
            
        # Boundary contributions
        b_weights = np.exp(-alpha * b_terms)
        # grad[k] += b_weights[k] * alpha * c[k] / ||c[k]||
        norms = np.linalg.norm(c, axis=1)
        norms[norms < 1e-6] = 1e-6
        grad += (b_weights * alpha / norms[:, None]) * c
        
        # Final gradient scaling
        grad = (-1.0 / (alpha * sum_exp)) * grad
        
        return obj, grad.flatten()

    # Optimization
    # We might need multiple restarts or alpha annealing.
    # Given time, we do one strong run.
    
    best_x = x0
    best_score = -1e10
    
    # Helper to compute actual score
    def get_score(x):
        c = x.reshape(N, 2)
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=-1)
        i_idx, j_idx = np.triu_indices(N, k=1)
        min_pair = np.min(d[i_idx, j_idx]) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(c, axis=1))
        return min(min_pair, min_bound)
    
    # Run optimization
    # L-BFGS-B
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
        
        if res.success or res.fun < best_score:
            best_x = res.x
            best_score = get_score(best_x)
            
            # Check time
            if time.time() > deadline:
                return best_x.reshape(N, 2)
                
            # Try a second phase with higher alpha for refinement?
            # The objective function uses fixed alpha.
            # We can just re-run with the same function but tighter tolerances if time permits.
            # Or, simply return.
            
    except Exception:
        pass
    
    # Final check
    return best_x.reshape(N, 2)
