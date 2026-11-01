"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_score(centers):
    """Compute the maximum common radius for the given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Min distance between pairs
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Min distance to boundary
    min_boundary = np.inf
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centers[i][k])
            if d < min_boundary:
                min_boundary = d
    
    return min(min_pair_dist, min_boundary) / 2.0


def _soft_min_score(centers, temperature=0.01):
    """Smooth approximation of the score using log-sum-exp."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Compute all pairwise distances
    pair_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            pair_dists.append(d)
    
    # Compute boundary distances
    boundary_dists = []
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centers[i][k])
            boundary_dists.append(d)
    
    all_dists = np.array(pair_dists + boundary_dists)
    
    # Soft min using log-sum-exp
    min_dist = np.logsumexp(-all_dists / temperature) * (-temperature)
    
    return min_dist / 2.0


def _soft_min_grad(centers, temperature=0.01):
    """Gradient of the soft min score."""
    n = len(centers)
    if n == 0:
        return np.zeros_like(centers)
    
    grad = np.zeros_like(centers)
    
    # Pairwise distances
    weights_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue
            w = np.exp(-d / temperature)
            weights_sum += w
            grad[i] += w * diff / d
            grad[j] -= w * diff / d
    
    # Boundary distances
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centers[i][k])
            if d < 1e-10:
                continue
            w = np.exp(-d / temperature)
            weights_sum += w
            if centers[i][k] >= 0:
                grad[i][k] += w
            else:
                grad[i][k] -= w
    
    # Normalize and negate (we want to maximize, so minimize negative)
    if weights_sum > 0:
        grad /= weights_sum
        grad *= -temperature
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Try multiple temperature values for soft-min
    temperatures = [0.05, 0.03, 0.02, 0.01, 0.005]
    
    for temp in temperatures:
        if time.time() > deadline - 20:
            break
        
        # Define objective and gradient functions
        def objective(x):
            centers_2d = x.reshape(-1, 3)
            return -_soft_min_score(centers_2d, temperature=temp)
        
        def gradient(x):
            centers_2d = x.reshape(-1, 3)
            return _soft_min_grad(centers_2d, temperature=temp).reshape(-1)
        
        # Optimize
        x0 = centers.flatten()
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if result.success or result.fun < objective(x0):
                new_centers = result.x.reshape(-1, 3)
                # Clip to valid range
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_score = _compute_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    centers = new_centers.copy()
                    
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        temp_final = 0.001
        
        def objective_final(x):
            centers_2d = x.reshape(-1, 3)
            return -_soft_min_score(centers_2d, temperature=temp_final)
        
        def gradient_final(x):
            centers_2d = x.reshape(-1, 3)
            return _soft_min_grad(centers_2d, temperature=temp_final).reshape(-1)
        
        x0 = best_centers.flatten()
        try:
            result = minimize(
                objective_final,
                x0,
                jac=gradient_final,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success or result.fun < objective_final(x0):
                new_centers = result.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_score = _compute_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    
        except Exception as e:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
