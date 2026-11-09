"""Model-and-optimize approach for covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _covering_radius_and_worst_facets(points, top_k=1):
    """
    Compute the covering radius in radians and identify the worst facets.
    Returns (radius, list of (facet_indices, circumcenter) for top_k worst facets).
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, []
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, []
    facets = hull.simplices  # (M, 3) array of indices
    
    # Compute circumcenter and cos_r for each facet
    facet_data = []
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        # Compute circumcenter direction
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_r = np.clip(p @ a, -1.0, 1.0)
        facet_data.append((cos_r, facet.copy(), p.copy()))
    
    if not facet_data:
        return np.pi, []
    
    # Sort by cos_r (ascending = largest radius first)
    facet_data.sort(key=lambda x: x[0])
    
    min_cos = facet_data[0][0]
    radius = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    # Return top_k worst facets
    top_facets = [(f[1], f[2]) for f in facet_data[:top_k]]
    
    return radius, top_facets


def _soft_covering_radius_and_grad(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    This creates a differentiable approximation of the max over facets.
    Returns (soft_radius, gradient)
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    m = len(facets)
    if m == 0:
        return np.pi, np.zeros_like(points)
    
    # Compute cos_r for each facet
    cos_vals = np.zeros(m)
    valid = np.ones(m, dtype=bool)
    facet_points = np.zeros((m, 3, 3))
    
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            valid[i] = False
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_vals[i] = np.clip(p @ a, -1.0, 1.0)
        facet_points[i] = np.array([a, b, c])
    
    if not np.any(valid):
        return np.pi, np.zeros_like(points)
    
    # Use log-sum-exp for smooth max
    # We want to minimize the max angular distance = arccos(min cos_r)
    # Equivalent to maximizing min cos_r
    # So we want a soft-min of cos_vals
    valid_cos = cos_vals[valid]
    
    # Soft-min: -logsumexp(-x) / beta
    shifted = -valid_cos / beta
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_cos = -beta * log_sum_exp
    
    # Gradient with respect to points
    # d(soft_min)/d(cos_i) = -exp(-cos_i/beta) / sum(exp(-cos_j/beta))
    weights = np.exp(-valid_cos / beta)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    for i in range(m):
        if not valid[i]:
            continue
        w = weights[i]
        facet = facets[i]
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            continue
        p = p / p_norm
        if p @ a < 0:
            p = -p
        
        # d(cos_r)/d(points) for this facet
        # cos_r = p · a, but p depends on a, b, c
        # Simplified gradient: approximate as moving points toward p
        # This is a heuristic gradient
        grad[facet[0]] += w * (p - a)
        grad[facet[1]] += w * (p - b)
        grad[facet[2]] += w * (p - c)
    
    # Soft radius in radians
    soft_cos = np.clip(soft_min_cos, -1.0, 1.0)
    soft_radius = np.arccos(soft_cos)
    
    return soft_radius, grad


def _local_optimize(points, max_iters=5000, deadline=None):
    """
    Combined local optimization using geometric repair with adaptive step size.
    """
    points = points.copy()
    n = points.shape[0]
    best_radius = _covering_radius_and_worst_facets(points)[0]
    no_improve = 0
    step = 0.08
    min_step = 1e-8
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=3)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-12:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive step size: reduce more aggressively when stuck
        if no_improve > 50:
            step *= 0.7
            no_improve = 0
        elif no_improve > 20:
            step *= 0.85
        elif no_improve < 10 and step < 0.1:
            step = min(step * 1.1, 0.15)
        
        if step < min_step:
            break
        
        # Move vertices of top worst facets toward their circumcenters
        moved_indices = set()
        for facet, circ in top_facets:
            for idx in facet:
                if idx in moved_indices:
                    continue
                moved_indices.add(idx)
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    new_pt = pt + step * direction
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    points[idx] = new_pt
    
    return points


def _smooth_optimize(points, max_iters=200, deadline=None):
    """
    Smooth optimization using log-sum-exp surrogate to navigate non-smooth landscape.
    """
    points = points.copy().flatten()
    n = N
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(n, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        soft_r, grad = _soft_covering_radius_and_grad(pts, beta=30.0)
        
        # Project gradient back to tangent space
        grad_proj = np.zeros_like(pts)
        for i in range(n):
            grad_proj[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
        
        return soft_r, grad_proj.flatten()
    
    current = points.flatten()
    best_radius = _covering_radius_and_worst_facets(points)[0]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        val, grad = objective_and_grad(current)
        
        if not np.isfinite(val):
            break
        
        # Simple gradient descent with adaptive step
        step_size = 0.005
        for ls_iter in range(10):
            new_flat = current - step_size * grad
            new_pts = new_flat.reshape(n, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_pts = new_pts / norms
            
            new_radius, _ = _covering_radius_and_worst_facets(new_pts)
            
            if new_radius < best_radius - 1e-12:
                current = new_flat
                points = new_pts
                best_radius = new_radius
                break
            else:
                step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return points


def _polish(points, max_iters=2000, deadline=None):
    """
    High-precision polish pass with very small steps to fine-tune the configuration.
    """
    points = points.copy()
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.005
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=2)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-14:
            best_radius = radius
            # Reduce step slightly on improvement
            step *= 0.97
        else:
            step *= 0.5
        
        if step < 1e-12:
            break
        
        # Move vertices of worst facets toward their circumcenters
        moved_indices = set()
        for facet, circ in top_facets:
            for idx in facet:
                if idx in moved_indices:
                    continue
                moved_indices.add(idx)
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    new_pt = pt + step * direction
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with random initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 35.0:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random points
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturbed best or random
            if best_points is not None and rng.random() < 0.6:
                perturbation = 0.2 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization with geometric repair
        remaining_time = deadline - time.time() - 15.0
        if remaining_time < 15.0:
            break
        
        max_iters = min(4000, int(remaining_time * 25))  # Rough estimate
        optimized = _local_optimize(points, max_iters=max_iters, deadline=deadline)
        
        # Smooth optimization phase if we have time
        if time.time() < deadline - 12.0:
            optimized = _smooth_optimize(optimized, max_iters=100, deadline=deadline)
        
        # Polish pass if we have time
        if time.time() < deadline - 8.0:
            optimized = _polish(optimized, max_iters=1500, deadline=deadline)
        
        radius, _ = _covering_radius_and_worst_facets(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _polish(best_points, max_iters=1000, deadline=deadline)
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
    
    if best_points is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points
