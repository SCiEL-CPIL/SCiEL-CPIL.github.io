"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _soft_covering_radius_and_grad(points, beta=100.0):
    """
    Compute soft covering radius and gradient using log-sum-exp over facet circumradii.
    Returns (soft_radius, gradient of soft_radius w.r.t. points).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi, np.zeros_like(points)
    
    # Compute circumradius (cosine of angle from origin to circumcenter) for each facet
    # For a facet with vertices a, b, c, the circumcenter direction is:
    # p = normalize(cross(a-b, b-c))
    # cos_val = p @ a (which equals p @ b = p @ c)
    # The angular distance is arccos(cos_val)
    
    cos_vals = np.zeros(len(facets))
    normals = np.zeros((len(facets), 3))
    valid = np.zeros(len(facets), dtype=bool)
    
    for i, facet in enumerate(facets):
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
        
        cos_val = p @ a
        cos_vals[i] = cos_val
        normals[i] = p
        valid[i] = True
    
    if not np.any(valid):
        return np.pi, np.zeros_like(points)
    
    # Soft max of angular distances = soft min of cos_vals
    # We want to minimize max(arccos(cos_val_i)) = -min(arccos(cos_val_i))... 
    # Actually, larger angular distance = smaller cos_val.
    # So we want to minimize the maximum angular distance.
    # Let theta_i = arccos(cos_val_i). We want min max theta_i.
    # Equivalently, we can work with the cos_vals directly.
    # max theta_i = arccos(min cos_val_i)
    # So we want to maximize min cos_val_i.
    # Soft max of theta = soft min of cos... this is getting complicated.
    # Let's just use soft max of theta directly.
    
    # theta_i = arccos(cos_vals[i])
    # We want soft max of theta_i
    # soft_max(theta) = (1/beta) * log(sum(exp(beta * theta_i)))
    
    # But theta_i = arccos(cos_vals[i]) is not differentiable in a simple way.
    # Alternative: work with f_i = -cos_vals[i]. We want to maximize min cos_vals[i],
    # which is equivalent to minimizing max(-cos_vals[i]).
    # soft_max(-cos_vals) = (1/beta) * log(sum(exp(-beta * cos_vals)))
    # Gradient of this w.r.t. cos_vals: -exp(-beta*cos_vals) / sum(exp(-beta*cos_vals))
    # Then chain rule to get gradient w.r.t. points.
    
    # Let's use this approach: minimize soft_max(-cos_vals)
    # This is a smooth surrogate for max angular distance (since arccos is monotone decreasing).
    
    neg_cos = -cos_vals[valid]
    # Clip to avoid overflow
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    soft_val = (1.0 / beta) * (max_neg + np.log(sum_exp))
    
    # Gradient w.r.t. cos_vals: -exp(-beta*cos) / sum(exp(-beta*cos)) = -exp_vals / sum_exp
    grad_cos = np.zeros(len(facets))
    grad_cos[valid] = -exp_vals / sum_exp
    
    # Now compute gradient w.r.t. points
    # For each facet, cos_val = p @ a where p = normalize(cross(a-b, b-c))
    # This is complex. Let's use numerical gradient or a simpler approach.
    
    # Alternative: use the fact that for a fixed facet, the circumcenter direction p
    # depends on the three points. The derivative is complex.
    
    # Simpler approach: use finite differences for the gradient of the soft objective.
    # But that's expensive. Let's instead just use the worst facets approach but with
    # a smooth weighting.
    
    # For now, return the soft value and zero gradient, and use it for ranking.
    # The actual optimization will use the worst facets with smooth weights.
    
    return soft_val, grad_cos, normals, valid

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi
    
    min_cos = 1.0
    for facet in facets:
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
        
        cos_val = p @ a
        if cos_val < min_cos:
            min_cos = cos_val
    
    return np.arccos(np.clip(min_cos, -1.0, 1.0))

def _weighted_worst_facets(points, k=20, beta=50.0):
    """
    Return weighted worst facets using soft-max weighting.
    Returns list of (indices, circumcenter, weight) for top k facets.
    """
    points = np.asarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    facets = hull.simplices
    if len(facets) == 0:
        return []
    
    cos_vals = np.zeros(len(facets))
    normals = np.zeros((len(facets), 3))
    valid = np.zeros(len(facets), dtype=bool)
    
    for i, facet in enumerate(facets):
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
        
        cos_val = p @ a
        cos_vals[i] = cos_val
        normals[i] = p
        valid[i] = True
    
    if not np.any(valid):
        return []
    
    # Soft-max weighting: weight_i = exp(-beta * cos_val_i) / sum(exp(-beta * cos_val_j))
    # Larger weight for smaller cos_val (larger angular distance)
    neg_cos = -cos_vals[valid]
    neg_cos_clipped = np.clip(neg_cos, -50.0, 50.0)
    max_neg = np.max(neg_cos_clipped)
    exp_vals = np.exp(beta * (neg_cos_clipped - max_neg))
    sum_exp = np.sum(exp_vals)
    weights = np.zeros(len(facets))
    weights[valid] = exp_vals / sum_exp
    
    # Get top k by weight
    valid_indices = np.where(valid)[0]
    sorted_idx = valid_indices[np.argsort(weights[valid_indices])[::-1]]
    
    result = []
    for idx in sorted_idx[:k]:
        if weights[idx] < 1e-8:
            break
        result.append((facets[idx].copy(), normals[idx].copy(), weights[idx]))
    
    return result


def _worst_facets(points, k=5):
    """Return the k worst facets (by circumradius) as list of (indices, circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    facets = hull.simplices
    facet_data = []
    for facet in facets:
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
        
        cos_val = p @ a
        facet_data.append((cos_val, facet.copy(), p.copy()))
    
    if not facet_data:
        return []
    
    facet_data.sort(key=lambda x: x[0])
    return [(f[1], f[2]) for f in facet_data[:k]]


def _gradient_descent(points, deadline, max_iters=3000, base_step=0.05):
    """Optimize points using gradient descent toward circumcenters of worst facets with soft weighting."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    beta = 30.0  # Start with moderate softness
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.85
        
        if step < 1e-10:
            break
        
        # Increase beta over time to focus more on worst facets
        progress = it / max_iters
        beta = 30.0 + 70.0 * progress
        
        worst = _weighted_worst_facets(points, k=15, beta=beta)
        if not worst:
            break
        
        # Accumulate displacement for each point
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                # Move toward circumcenter but stay on sphere
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    # Tangent component
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        # Apply displacement
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def _random_restart_optimize(rng, deadline, max_iters=2000):
    """Generate random initial config and optimize."""
    points = rng.standard_normal((N, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return _gradient_descent(points, deadline, max_iters=max_iters)

def _polish(points, deadline, max_iters=500, base_step=0.005):
    """Final high-precision polish with very small steps."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.02, 0.01)
        else:
            step *= 0.9
        
        if step < 1e-12:
            break
        
        # Use only the single worst facet with high beta
        worst = _weighted_worst_facets(points, k=3, beta=100.0)
        if not worst:
            break
        
        displacement = np.zeros_like(points)
        for facet, circ, weight in worst:
            for idx in facet:
                pt = points[idx]
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    tangent = direction - np.dot(direction, pt) * pt
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        tangent = tangent / norm_t
                        displacement[idx] += weight * tangent
        
        for idx in range(N):
            if np.linalg.norm(displacement[idx]) > 1e-10:
                new_pt = points[idx] + step * displacement[idx]
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[idx] = new_pt
    
    return points


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = fib_points / np.linalg.norm(fib_points, axis=1, keepdims=True)
    
    best_points = _gradient_descent(fib_points, deadline, max_iters=2000, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        # Mix: sometimes perturb best, sometimes random
        if best_points is not None and restart % 2 == 0:
            perturbation = 0.15 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(2000, int(remaining * 20))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.06)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, deadline, max_iters=800, base_step=0.003)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
