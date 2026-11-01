import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute pairwise distances efficiently
    # Using upper triangle indices
    iu = np.triu_indices(n, k=1)
    vecs = points[iu[0]] - points[iu[1]]
    dists = np.sqrt(np.sum(vecs**2, axis=1))
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    points = []
    # Estimate spacing.
    # Area per circle ~ pi * r^2.
    # Hex packing density ~ pi / (2 * sqrt(3)) * (2r)^2 ? No.
    # Area of hexagon with side s (distance between centers is s? No, s is usually side length).
    # Distance between centers in hex lattice is d.
    # Area per point = d^2 * sqrt(3) / 2.
    # N * d^2 * sqrt(3) / 2 ~ pi (area of unit circle).
    # d ~ sqrt(2 * pi / (N * sqrt(3))).
    d = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    # x = d * (i + 0.5 * (j % 2))
    # y = d * j * sqrt(3) / 2
    max_k = int(1.0 / (d * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = d * (i + 0.5 * (j % 2))
            y = d * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # If we didn't generate enough, shrink spacing slightly and retry
        d *= 0.9
        max_k = int(1.0 / (d * 0.5)) + 3
        candidates = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = d * (i + 0.5 * (j % 2))
                y = d * j * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: random points
        points = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1.0
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.9

    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    
    return points

def _local_repair(points, num_iters=5, step_size=0.01):
    """
    Simple local repair: find the most conflicting pairs and push them apart.
    """
    points = points.copy()
    iu = np.triu_indices(len(points), k=1)
    
    for _ in range(num_iters):
        # Calculate all pair distances
        vecs = points[iu[0]] - points[iu[1]]
        dists = np.sqrt(np.sum(vecs**2, axis=1))
        
        # Find the minimum distance
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        if min_dist > 0.001: # If minimum distance is large enough, stop
            break
            
        i, j = iu[0][min_idx], iu[1][min_idx]
        # Push i and j apart
        if min_dist > 1e-9:
            direction = vecs[min_idx] / min_dist
        else:
            direction = np.array([1.0, 0.0])
            
        points[i] += direction * step_size * 0.5
        points[j] -= direction * step_size * 0.5
        
        # Also check boundaries for these points
        for idx in [i, j]:
            norm = np.linalg.norm(points[idx])
            if norm > 0.95:
                points[idx] /= norm
                points[idx] *= 0.95

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)
    
    # Initialization
    points = _hex_lattice_init(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Optimization using L-BFGS-B with smooth soft-min objective
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate pair distances efficiently
        vecs = p[i_idx] - p[j_idx]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min calculation
        # z = -beta * c
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        if sum_exp == 0:
             return 0.0, np.zeros_like(p)
             
        log_sum_exp = max_z + np.log(sum_exp)
        
        # Objective value: -soft_min = 1/beta * log_sum_exp
        val = log_sum_exp / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        # Pair contributions
        # d(pair_dist)/dx_i = 0.5 * (x_i - x_j) / ||x_i - x_j||
        # d(pair_dist)/dx_j = 0.5 * (x_j - x_i) / ||x_i - x_j||
        
        safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        
        # coeff for x_i: w * 0.5 / dist
        # coeff for x_j: w * 0.5 / dist
        coeff = w_pairs * 0.5 / safe_dists
        
        # Vector from j to i
        # grad_x_i += coeff * vec
        # grad_x_j -= coeff * vec
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        # Boundary contributions
        # d(bound_dist)/dx_i = -x_i / ||x_i||
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, grad

    # Optimization loop with increasing beta
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() > deadline - 5:
                break
                
            candidate = res.x.reshape(-1, 2)
            
            # Apply local repair to break local minima
            if time.time() < deadline - 10:
                 candidate = _local_repair(candidate, num_iters=10, step_size=0.005)
            
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
                current_points = candidate
            else:
                # If it didn't improve, keep the previous best as the start for next beta
                current_points = best_points
                
        except Exception:
            pass

    # Final polish with very high beta if time permits
    if time.time() < deadline - 5:
        try:
            beta = 2000.0
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                best_points.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 50,
                    'gtol': 1e-10,
                    'ftol': 1e-13,
                    'maxls': 10,
                    'disp': False
                }
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > best_score:
                best_points = candidate
                best_score = candidate_score
        except Exception:
            pass

    # Ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms > 1.0
    if np.any(outside):
        best_points[outside] /= norms[outside, np.newaxis]
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
