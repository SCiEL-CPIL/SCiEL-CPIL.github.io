"""Improved program: ring-based initialization + scipy constrained optimization for circle packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _ring_initialization(n):
    """
    Initialize points in concentric rings with hexagonal-like spacing.
    This respects the circular symmetry of the problem.
    """
    points = []
    rng = np.random.RandomState(42)
    
    # Start with a center point
    points.append([0.0, 0.0])
    
    # Add rings with increasing radius
    ring_radius = 0.05
    ring_spacing = 0.055  # approximate radius of each circle
    max_radius = 1.0
    
    while len(points) < n and ring_radius < max_radius:
        # Number of points in this ring
        circumference = 2 * np.pi * ring_radius
        n_in_ring = max(1, int(circumference / ring_spacing))
        
        # Don't exceed remaining points needed
        remaining = n - len(points)
        n_in_ring = min(n_in_ring, remaining)
        
        if n_in_ring <= 0:
            break
            
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for angle in angles:
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            # Check if inside unit circle
            if np.sqrt(x**2 + y**2) <= 1.0:
                points.append([x, y])
        
        ring_radius += ring_spacing * 0.866  # hexagonal spacing
    
    # If we don't have enough, add more rings or adjust
    while len(points) < n:
        # Add points in a spiral pattern
        idx = len(points)
        angle = idx * 0.1  # golden angle approximation
        radius = 0.03 * np.sqrt(idx)
        if radius > 1.0:
            radius = 1.0 - 0.01 * (idx - len(points))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        if np.sqrt(x**2 + y**2) <= 1.0:
            points.append([x, y])
    
    points = np.array(points[:n])
    
    # Add small random perturbation
    points = points + rng.normal(0, 0.002, size=points.shape)
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] = points[mask] / norms[mask][:, np.newaxis] * 0.99
    
    return points

def _objective_and_grad(x):
    """
    Objective: maximize the minimum radius (negated for minimization).
    Uses a soft-min approach for differentiability.
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Pairwise constraints: dists[i,j] / 2
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0
    
    # Boundary constraints
    norms = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft min with temperature
    tau = 50.0
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    log_sum_exp = np.log(np.sum(np.exp(-tau * shifted)))
    soft_min_val = min_val - (1.0 / tau) * log_sum_exp
    
    # Return negative for minimization
    return -soft_min_val

def _local_optimize_scipy(points, max_iter=200, time_limit=30.0):
    """
    Local optimization using scipy's SLSQP with time limit.
    """
    start_time = time.time()
    x0 = points.flatten()
    
    # Constraints: all points must be inside unit circle
    def boundary_constraint(x):
        pts = x.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        return 1.0 - norms
    
    constraints = [{'type': 'ineq', 'fun': boundary_constraint}]
    
    # Bounds: keep points within [-1, 1] in each dimension
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_result = None
    best_score = _compute_score(points)
    best_points = points.copy()
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        
        if result.success or (result.fun < -best_score + 1e-10):
            result_points = result.x.reshape(-1, 2)
            result_score = _compute_score(result_points)
            if result_score > best_score:
                best_score = result_score
                best_points = result_points.copy()
                best_result = result
        
    except Exception as e:
        pass
    
    return best_points, best_score

def _polish_with_newton(points, max_iter=50, time_limit=20.0):
    """
    Final polish using a more aggressive soft-min with higher temperature decay.
    """
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _compute_score(points)
    
    # Start with moderate temperature and decay
    tau_start = 30.0
    tau_end = 200.0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Interpolate temperature
        frac = it / max_iter
        tau = tau_start + (tau_end - tau_start) * frac
        
        # Compute soft-min and gradient
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu] / 2.0
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        min_val = np.min(all_dists)
        shifted = all_dists - min_val
        log_sum_exp = np.log(np.sum(np.exp(-tau * shifted)))
        soft_min_val = min_val - (1.0 / tau) * log_sum_exp
        
        # Weights
        exp_terms = np.exp(-tau * shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient
        grad = np.zeros_like(current)
        
        i_idx, j_idx = iu
        w_pairs = weights[:len(pair_dists)]
        d_pairs = 2.0 * pair_dists
        mask = d_pairs > 1e-12
        w_pairs_safe = np.where(mask, w_pairs, 0.0)
        d_pairs_safe = np.where(mask, d_pairs, 1.0)
        
        diff_vec = current[i_idx] - current[j_idx]
        unit_vecs = diff_vec / d_pairs_safe[:, np.newaxis]
        coeffs = w_pairs_safe / 2.0
        
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * unit_vecs)
        
        w_bounds = weights[len(pair_dists):]
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        bound_grads = -current / norms_safe[:, np.newaxis]
        grad += w_bounds[:, np.newaxis] * bound_grads
        
        # Adaptive step size
        step_size = 0.001 * (1.0 - frac * 0.8)
        current = current - step_size * grad
        
        # Check score
        score = _compute_score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Ring-based initialization
    points = _ring_initialization(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Scipy constrained optimization
    if time.time() < deadline - 40:
        points, score = _local_optimize_scipy(points, max_iter=100, time_limit=30.0)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 3. Polish with Newton-like method
    if time.time() < deadline - 15:
        points, score = _polish_with_newton(points, max_iter=50, time_limit=15.0)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # 4. Final high-temperature polish
    if time.time() < deadline - 5:
        points, score = _polish_with_newton(points, max_iter=30, time_limit=5.0)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final check: ensure points are inside
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask][:, np.newaxis] * 0.999
    
    return best_points
