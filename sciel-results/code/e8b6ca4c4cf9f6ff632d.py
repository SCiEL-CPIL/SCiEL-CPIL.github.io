"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 50
DEADLINE = 110.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    # Use a smaller spacing to ensure we have enough points
    # Start with a reasonable estimate
    r_est = 0.12  # Known packing has r ~ 0.1258
    s = 2 * r_est
    
    points = []
    # Generate a grid large enough
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:  # Keep inside with margin
                points.append([x, y])
                
    if len(points) == 0:
        # Fallback: random points
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # If we don't have enough, generate more randomly
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    # Sort by distance to origin to pick the most central ones
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    # To avoid overflow, subtract max
    v = np.array(values)
    m = np.min(v)
    # If beta is large, this approximates min(v)
    # We want to maximize min(v), so we minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * v)))
    # This is numerically stable if we do:
    # exp(-beta * (v - m))
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    constraints: 
      1. pairwise distances / 2
      2. 1 - norm(center)
    """
    centers = params.reshape(n, 2)
    
    # Calculate pairwise distances using vectorized operations
    # diff[i, j, :] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get upper triangle indices (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Constraints values
    # 1. d_ij / 2
    vals = np.concatenate([pair_dists / 2.0, 1.0 - np.linalg.norm(centers, axis=1)])
    
    # Soft min
    obj = -_soft_min(vals, beta)
    
    # Gradient calculation
    exp_vals = np.exp(-beta * (vals - np.min(vals)))
    S = np.sum(exp_vals)
    weights = exp_vals / S  # Weights for each constraint
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    # Bound contributions
    bound_weights = weights[n*(n-1)//2:]
    norms = np.linalg.norm(centers, axis=1)
    for i in range(n):
        w = bound_weights[i]
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
        
    return obj, -grad

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Minimum distance to boundary
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Initial guess
    init_centers = _get_initial_hex_lattice(N)
    
    # Validate initial
    init_score = _compute_score(init_centers)
    if np.all(np.isfinite(init_centers)) and init_score > 0:
        best_score = init_score
        best_centers = init_centers.copy()
    
    # Run optimizations with different beta values
    # Start with moderate beta, then increase
    betas = [20.0, 50.0, 100.0]
    
    rng = np.random.default_rng(42)
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 10:  # Leave 10s margin
            break
        
        # Use current best as starting point, add small noise
        if best_centers is not None:
            noisy_centers = best_centers + rng.normal(0, 0.02, best_centers.shape)
        else:
            noisy_centers = init_centers + rng.normal(0, 0.05, init_centers.shape)
        
        # Ensure inside circle
        norms = np.linalg.norm(noisy_centers, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            noisy_centers[mask] *= 0.95 / norms[mask]
        
        # Optimize
        try:
            res = minimize(
                _objective_and_gradient, 
                noisy_centers.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-5}
            )
        except Exception:
            continue
            
        centers = res.x.reshape(N, 2)
        
        # Check validity
        if np.any(~np.isfinite(centers)):
            continue
            
        # Calculate actual score
        r = _compute_score(centers)
        
        if r > best_score:
            best_score = r
            best_centers = centers.copy()
            
            # If good, try polishing with higher beta
            if r > 0.12 and time.time() - start_time < DEADLINE - 20:
                for beta_polish in [100.0, 200.0]:
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    try:
                        res_polish = minimize(
                            _objective_and_gradient, 
                            centers.flatten(), 
                            args=(N, beta_polish),
                            method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': 50, 'gtol': 1e-6}
                        )
                        centers_polish = res_polish.x.reshape(N, 2)
                        
                        if np.any(~np.isfinite(centers_polish)):
                            continue
                            
                        r_p = _compute_score(centers_polish)
                        if r_p > best_score:
                            best_score = r_p
                            best_centers = centers_polish.copy()
                    except Exception:
                        pass

    if best_centers is None:
        # Fallback to initial
        return init_centers
        
    return best_centers
# EVOLVE-BLOCK-END
