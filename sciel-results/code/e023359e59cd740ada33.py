"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the actual radius: min of half-pair-distances and boundary distances."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary distances
    boundary = np.min(np.minimum(0.5 - np.abs(centers), 0.5))
    return min(min_pair, boundary)

def _soft_radius(centers, beta=50.0):
    """Smooth surrogate for the radius using log-sum-exp on negative distances."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft-min of distances (using log-sum-exp trick for stability)
    # We want soft_min(dists) = -logsumexp(-dists/beta) * beta
    # But dists can be zero, so use dists_sq for stability
    # soft_min of dists ≈ sqrt(soft_min of dists_sq) is not quite right
    # Better: soft_min of dists = -beta * log(mean(exp(-dists/beta)))
    # For numerical stability with potentially large beta:
    half_dists = np.sqrt(dists_sq)
    half_dists = np.where(np.isinf(half_dists), 1e10, half_dists)
    
    # Soft-min of pair distances
    neg_dists = -half_dists / beta
    max_neg = np.max(neg_dists, axis=(0,1))
    if np.isfinite(max_neg):
        log_sum = np.log(np.sum(np.exp(neg_dists - max_neg))) + max_neg
        soft_min_pairs = -beta * log_sum
    else:
        soft_min_pairs = 0.0
    
    # Soft-min of boundary distances
    boundary_dists = np.minimum(0.5 - np.abs(centers), 0.5)
    neg_bound = -boundary_dists / beta
    max_bound = np.max(neg_bound)
    if np.isfinite(max_bound):
        log_sum_b = np.log(np.sum(np.exp(neg_bound - max_bound))) + max_bound
        soft_min_bound = -beta * log_sum_b
    else:
        soft_min_bound = 0.0
    
    # Combine: soft-min of (soft_min_pairs, soft_min_bound)
    # Use another soft-min
    vals = np.array([soft_min_pairs, soft_min_bound])
    neg_vals = -vals / beta
    max_v = np.max(neg_vals)
    log_sum_v = np.log(np.sum(np.exp(neg_vals - max_v))) + max_v
    return -beta * log_sum_v

def _negative_soft_radius(centers, beta=50.0):
    """Negative of soft radius for minimization."""
    return -_soft_radius(centers, beta)

def _init_grid(n=N):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n].astype(float)

def _init_random(n=N, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    # Initialize in a grid-like pattern with some randomness
    m = int(np.ceil(np.sqrt(n)))
    base = _init_grid(n)
    noise = rng.uniform(-0.02, 0.02, size=base.shape)
    centers = base + noise
    # Clip to stay inside
    centers[:, 0] = np.clip(centers[:, 0], -0.48, 0.48)
    centers[:, 1] = np.clip(centers[:, 1], -0.48, 0.48)
    return centers

def _optimize_from(centers, deadline, beta_schedule=None):
    """Optimize from given centers, return best centers found."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    if beta_schedule is None:
        beta_schedule = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 5.0:
            break
        
        def objective(x):
            c = x.reshape(-1, 2)
            return _negative_soft_radius(c, beta)
        
        def gradient(x):
            c = x.reshape(-1, 2)
            n = len(c)
            grad = np.zeros_like(c)
            
            # Gradient of soft-min of pair distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            half_dists = np.sqrt(dists_sq)
            half_dists = np.where(np.isinf(half_dists), 1e10, half_dists)
            
            neg_dists = -half_dists / beta
            max_neg = np.max(neg_dists, axis=(0,1))
            if np.isfinite(max_neg):
                exp_vals = np.exp(neg_dists - max_neg)
                # Set diagonal to 0 for gradient
                np.fill_diagonal(exp_vals, 0)
                weights = exp_vals / np.sum(exp_vals)
                
                # d(soft_min)/d(c_i) = -sum_j weights[i,j] * (c_i - c_j) / half_dists[i,j]
                for i in range(n):
                    for j in range(n):
                        if i != j and half_dists[i,j] > 1e-10:
                            grad[i] -= weights[i,j] * (c[i] - c[j]) / half_dists[i,j]
            
            # Gradient of soft-min of boundary distances
            boundary_dists = np.minimum(0.5 - np.abs(c), 0.5)
            neg_bound = -boundary_dists / beta
            max_bound = np.max(neg_bound)
            if np.isfinite(max_bound):
                exp_b = np.exp(neg_bound - max_bound)
                weights_b = exp_b / np.sum(exp_b)
                # d(0.5 - |x|)/dx = -sign(x) if x>0, sign(x) if x<0, 0 if x=0
                # d(soft_min)/dc = -sum weights * d(boundary_dist)/dc
                for i in range(2):
                    signs = np.sign(c[:, i])
                    signs[np.abs(c[:, i]) < 1e-10] = 0
                    grad[:, i] -= weights_b * signs * 1.0  # derivative of -|x| is -sign(x), but we have -boundary_dist
            
            # Combine gradients with soft-min weighting
            # Simplified: just use pair gradient as primary
            return grad.ravel()
        
        x0 = centers.ravel()
        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            new_centers = result.x.reshape(-1, 2)
            # Clip to valid range
            new_centers[:, 0] = np.clip(new_centers[:, 0], -0.4999, 0.4999)
            new_centers[:, 1] = np.clip(new_centers[:, 1], -0.4999, 0.4999)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
            centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = _init_grid()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        centers = _init_random(seed=i)
        opt_centers, opt_radius = _optimize_from(centers, deadline)
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Final polish with tighter tolerance if time allows
    if time.time() < deadline - 5.0:
        def obj_polish(x):
            c = x.reshape(-1, 2)
            return _negative_soft_radius(c, beta=1.0)
        
        def grad_polish(x):
            c = x.reshape(-1, 2)
            n = len(c)
            grad = np.zeros_like(c)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            half_dists = np.sqrt(dists_sq)
            half_dists = np.where(np.isinf(half_dists), 1e10, half_dists)
            
            beta = 1.0
            neg_dists = -half_dists / beta
            max_neg = np.max(neg_dists, axis=(0,1))
            if np.isfinite(max_neg):
                exp_vals = np.exp(neg_dists - max_neg)
                np.fill_diagonal(exp_vals, 0)
                weights = exp_vals / np.sum(exp_vals)
                for i in range(n):
                    for j in range(n):
                        if i != j and half_dists[i,j] > 1e-10:
                            grad[i] -= weights[i,j] * (c[i] - c[j]) / half_dists[i,j]
            
            boundary_dists = np.minimum(0.5 - np.abs(c), 0.5)
            neg_bound = -boundary_dists / beta
            max_bound = np.max(neg_bound)
            if np.isfinite(max_bound):
                exp_b = np.exp(neg_bound - max_bound)
                weights_b = exp_b / np.sum(exp_b)
                for i in range(2):
                    signs = np.sign(c[:, i])
                    signs[np.abs(c[:, i]) < 1e-10] = 0
                    grad[:, i] -= weights_b * signs
            
            return grad.ravel()
        
        try:
            result = minimize(obj_polish, best_centers.ravel(), method='L-BFGS-B', 
                             jac=grad_polish, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            polished = result.x.reshape(-1, 2)
            polished[:, 0] = np.clip(polished[:, 0], -0.4999, 0.4999)
            polished[:, 1] = np.clip(polished[:, 1], -0.4999, 0.4999)
            r = _compute_radius(polished)
            if r > best_radius:
                best_centers = polished
        except Exception:
            pass
    
    # Final validation and clipping
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.4999, 0.4999)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
