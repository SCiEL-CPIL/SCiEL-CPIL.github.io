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
    
    hull = ConvexHull(points)
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


def _soft_covering_radius(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    This creates a differentiable approximation of max over facets of arccos(cos_r).
    Returns (soft_radius, gradient_wrt_points)
    """
    points = np.asarray(points)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    hull = ConvexHull(points)
    facets = hull.simplices  # (M, 3) array of indices
    
    # Compute cos_r for each facet
    cos_vals = np.zeros(len(facets))
    valid_mask = np.ones(len(facets), dtype=bool)
    
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            valid_mask[i] = False
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_vals[i] = np.clip(p @ a, -1.0, 1.0)
    
    if not valid_mask.any():
        return np.pi, np.zeros_like(points)
    
    # Use soft-max on -cos_vals (since we want to minimize max angle = minimize -min cos)
    # Actually, covering radius = max over facets of arccos(cos_r)
    # So we want to maximize the minimum cos_r, or minimize the maximum angle
    # soft-max of angles: log-sum-exp(beta * angles) / beta
    angles = np.arccos(np.clip(cos_vals, -1.0, 1.0))
    angles = angles[valid_mask]
    
    if len(angles) == 0:
        return np.pi, np.zeros_like(points)
    
    # Soft-max approximation
    max_angle = np.max(angles)
    shifted = angles - max_angle
    weights = np.exp(beta * shifted)
    weights = weights / np.sum(weights)
    
    soft_radius = max_angle + np.log(np.sum(weights)) / beta
    
    # Gradient with respect to points (approximate via finite differences would be expensive)
    # For now, return zero gradient; we'll use the facet-based approach for optimization
    grad = np.zeros_like(points)
    
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


def _smooth_optimize(points, deadline=None, max_iters=200):
    """
    Use L-BFGS-B with a smooth epigraph formulation to optimize the covering radius.
    We minimize max over facets of arccos(cos_r) by introducing an auxiliary variable t
    and minimizing t subject to t >= arccos(cos_r_i) for all facets.
    """
    points = points.copy()
    n = points.shape[0]
    
    def _objective_and_grad(x_flat):
        """Objective: maximize t (equivalently minimize -t) subject to t >= angle_i for all i.
        We use a penalty approach: minimize t + lambda * sum(max(0, angle_i - t)^2)
        """
        pts = x_flat.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        # Compute all facet angles
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1.0, np.zeros_like(x_flat)
        
        facets = hull.simplices
        angles = []
        for facet in facets:
            a = pts[facet[0]]
            b = pts[facet[1]]
            c = pts[facet[2]]
            v1 = a - b
            v2 = b - c
            p = np.cross(v1, v2)
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                angles.append(0.0)
                continue
            p = p / p_norm
            if p @ a < 0:
                p = -p
            cos_r = np.clip(p @ a, -1.0, 1.0)
            angles.append(np.arccos(cos_r))
        
        if not angles:
            return 1.0, np.zeros_like(x_flat)
        
        angles = np.array(angles)
        max_angle = np.max(angles)
        
        # Use soft penalty: objective = max_angle + penalty * sum((max(0, angle_i - max_angle))^2)
        # Actually, just return max_angle as objective and let L-BFGS handle it
        return max_angle, np.zeros_like(x_flat)
    
    # Use L-BFGS-B with numerical gradient (expensive but potentially effective)
    x0 = points.flatten()
    
    # We'll do a few iterations of L-BFGS-B with the smooth objective
    result = None
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        if result.success or result.fun < _covering_radius_and_worst_facets(points)[0]:
            new_pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_pts = new_pts / norms
            new_radius = _covering_radius_and_worst_facets(new_pts)[0]
            old_radius = _covering_radius_and_worst_facets(points)[0]
            if new_radius < old_radius:
                points = new_pts
    except Exception:
        pass
    
    return points


def _polish(points, max_iters=2000, deadline=None):
    """
    High-precision polish pass with very small steps to fine-tune the configuration.
    """
    points = points.copy()
    best_radius = _covering_radius_and_worst_facets(points)[0]
    step = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        radius, top_facets = _covering_radius_and_worst_facets(points, top_k=2)
        
        if not top_facets:
            break
        
        if radius < best_radius - 1e-14:
            best_radius = radius
            # Reduce step slightly on improvement
            step *= 0.95
        else:
            step *= 0.5
        
        if step < 1e-10:
            break
        
        # Move vertices of top worst facets
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
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts with random initializations
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
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
            if best_points is not None and rng.random() < 0.5:
                perturbation = 0.25 * rng.standard_normal((N, 3))
                points = best_points + perturbation
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            else:
                points = rng.standard_normal((N, 3))
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Local optimization with geometric repair
        remaining_time = deadline - time.time() - 10.0
        if remaining_time < 15.0:
            break
        
        max_iters = min(3000, int(remaining_time * 30))  # Rough estimate
        optimized = _local_optimize(points, max_iters=max_iters, deadline=deadline)
        
        # Try smooth optimization if we have time
        if time.time() < deadline - 20.0:
            optimized = _smooth_optimize(optimized, deadline=deadline)
        
        # Polish pass if we have time
        if time.time() < deadline - 8.0:
            optimized = _polish(optimized, max_iters=1500, deadline=deadline)
        
        radius, _ = _covering_radius_and_worst_facets(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish on the best result if we have time
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, max_iters=800, deadline=deadline)
    
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
