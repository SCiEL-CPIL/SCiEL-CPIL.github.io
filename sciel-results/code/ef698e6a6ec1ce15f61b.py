"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _facet_circumcenters(points):
    """Compute circumcenter and cos value for each facet. Returns (facets, circumcenters, cos_values)."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return [], [], []
    
    facets = hull.simplices
    circumcenters = []
    cos_values = []
    
    for facet in facets:
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            circumcenters.append(np.zeros(3))
            cos_values.append(1.0)
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_val = p @ a
        circumcenters.append(p.copy())
        cos_values.append(cos_val)
    
    return facets, np.array(circumcenters), np.array(cos_values)


def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi
    
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return np.pi
    
    min_cos = np.min(cos_values)
    return np.arccos(np.clip(min_cos, -1.0, 1.0))


def _soft_covering_radius_and_grad(points, beta=100.0):
    """
    Compute soft covering radius and gradient using log-sum-exp over facet circumradii.
    Returns (soft_radius, gradient of soft_radius w.r.t. points).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return np.pi, np.zeros_like(points)
    
    # Use -cos as proxy for angular distance (since arccos is monotonic decreasing)
    # Soft max of angular distances = soft max of arccos(cos_val)
    # But arccos is expensive; use -cos_val directly as a smooth surrogate
    # Minimizing soft max of -cos_val is equivalent to maximizing soft min of cos_val
    
    neg_cos = -cos_values
    max_neg_cos = np.max(neg_cos)
    weights = np.exp(beta * (neg_cos - max_neg_cos))
    weights /= np.sum(weights)
    
    soft_radius = max_neg_cos + np.log(np.sum(weights)) / beta
    
    # Gradient: d(soft_radius)/d(points[i])
    # soft_radius depends on neg_cos through the log-sum-exp
    # d(soft_radius)/d(neg_cos_j) = weights[j]
    # d(neg_cos_j)/d(points[i]) = -d(cos_j)/d(points[i])
    # 
    # cos_j = p_j · a_j where p_j is the circumcenter (unit normal to facet j)
    # d(cos_j)/d(a_k) = p_j if k is in facet j, else 0
    # But p_j also depends on the points, so we need the full derivative
    # 
    # For simplicity, use the approximation that p_j is fixed (treat as given)
    # This is a common approximation in practice
    
    grad = np.zeros_like(points)
    for j, facet in enumerate(facets):
        w = weights[j]
        if w < 1e-15:
            continue
        p_j = circumcenters[j]
        for idx in facet:
            # d(cos_j)/d(points[idx]) ≈ p_j (approximation)
            # d(neg_cos_j)/d(points[idx]) = -p_j
            # d(soft_radius)/d(points[idx]) += w_j * (-p_j)
            # But we need to project to tangent space
            tangent = -p_j - np.dot(-p_j, points[idx]) * points[idx]
            grad[idx] += w * tangent
    
    return soft_radius, grad


def _gradient_descent(points, deadline, max_iters=3000, base_step=0.05):
    """Optimize points using gradient descent toward circumcenters of worst facets."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    patience = 0
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius - 1e-10:
            best_radius = radius
            patience = 0
            step = min(step * 1.03, 0.12)
        else:
            patience += 1
            step *= 0.9
            if patience > 50:
                break
        
        if step < 1e-12:
            break
        
        worst = _worst_facets(points, k=10)
        if not worst:
            break
        
        for facet, circ in worst:
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
                        new_pt = pt + step * tangent
                        new_pt = new_pt / np.linalg.norm(new_pt)
                        points[idx] = new_pt
    
    return points


def _worst_facets(points, k=5):
    """Return the k worst facets (by circumradius) as list of (indices, circumcenter)."""
    points = np.asarray(points, dtype=np.float64)
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return []
    
    order = np.argsort(cos_values)
    return [(facets[i], circumcenters[i]) for i in order[:k]]


def _lbfgs_smooth_optimize(points, deadline, max_iter=500):
    """Optimize using L-BFGS-B on the soft covering radius surrogate."""
    points_flat = points.flatten().copy()
    
    def fun(flat_pts):
        pts = flat_pts.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        soft_r, _ = _soft_covering_radius_and_grad(pts, beta=150.0)
        return soft_r

    def jac(flat_pts):
        pts = flat_pts.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        _, grad = _soft_covering_radius_and_grad(pts, beta=150.0)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad_tangent.flatten()
    
    result = minimize(fun, points_flat, jac=jac, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
    
    optimized = result.x.reshape(N, 3)
    norms = np.linalg.norm(optimized, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    optimized = optimized / norms
    
    return optimized


def _polish(points, deadline):
    """Final high-precision polish with multiple phases."""
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: Gradient descent with small steps
    if time.time() < deadline - 10.0:
        polished = _gradient_descent(points, deadline - 10.0, max_iters=500, base_step=0.01)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 2: L-BFGS on smooth surrogate
    if time.time() < deadline - 5.0:
        polished = _lbfgs_smooth_optimize(best_points, deadline - 5.0, max_iter=200)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 3: Very fine gradient descent
    if time.time() < deadline - 2.0:
        polished = _gradient_descent(best_points, deadline - 2.0, max_iters=300, base_step=0.003)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    return best_points


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
    
    # Initial optimization with gradient descent
    best_points = _gradient_descent(fib_points, deadline, max_iters=1500, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        # Mix: sometimes perturb best, sometimes random
        if best_points is not None and restart % 2 == 0:
            perturbation = 0.12 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(1500, int(remaining * 15))
        optimized = _gradient_descent(points, deadline, max_iters=max_iters, base_step=0.06)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _polish(best_points, deadline)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
