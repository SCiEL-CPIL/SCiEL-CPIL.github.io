"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

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


def _soft_covering_radius_and_grad(points, beta=50.0):
    """
    Compute a smooth approximation of the covering radius using soft-min over facet cosines.
    Returns (soft_radius, gradient w.r.t. points).
    
    The soft-min of cosines approximates the minimum cosine (which corresponds to max angle).
    We use: soft_min(c_i) ≈ -logsumexp(-beta * c_i) / beta
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    facets = hull.simplices
    if len(facets) == 0:
        return np.pi, np.zeros_like(points)
    
    # Compute cosines for all facets
    cos_vals = np.zeros(len(facets))
    normals = np.zeros((len(facets), 3))
    
    for i, facet in enumerate(facets):
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        v1 = a - b
        v2 = b - c
        p = np.cross(v1, v2)
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            cos_vals[i] = 1.0
            normals[i] = a
            continue
        p = p / p_norm
        
        if p @ a < 0:
            p = -p
        
        cos_vals[i] = p @ a
        normals[i] = p
    
    # Soft-min: -logsumexp(-beta * cos_vals) / beta
    neg_beta_cos = -beta * cos_vals
    max_neg_beta_cos = np.max(neg_beta_cos)
    log_sum_exp = max_neg_beta_cos + np.log(np.sum(np.exp(neg_beta_cos - max_neg_beta_cos)))
    soft_min_cos = -log_sum_exp / beta
    
    # Clip to valid range
    soft_min_cos = np.clip(soft_min_cos, -1.0, 1.0)
    soft_radius = np.arccos(soft_min_cos)
    
    # Gradient of soft_min_cos w.r.t. each point
    # d(soft_min_cos)/d(cos_i) = exp(-beta * cos_i) / sum(exp(-beta * cos_j))
    weights = np.exp(neg_beta_cos - max_neg_beta_cos)
    weights = weights / np.sum(weights)
    
    grad_points = np.zeros_like(points)
    
    for i, facet in enumerate(facets):
        w = weights[i]
        if w < 1e-12:
            continue
        
        a = points[facet[0]]
        b = points[facet[1]]
        c = points[facet[2]]
        
        # Gradient of cos_val w.r.t. a, b, c
        # cos_val = (n @ a) where n is the normalized normal
        # d(cos_val)/da = n, d(cos_val)/db = -n * (something), etc.
        # Actually, for a triangle (a,b,c), the normal n = cross(b-a, c-b)/|cross(b-a, c-b)|
        # This is complex. Let's use a simpler approach: 
        # The gradient of the covering radius w.r.t. point p_k is approximately
        # proportional to the normal of the worst facet containing p_k.
        
        n = normals[i]
        
        # For each vertex in the facet, the gradient contribution is -w * n
        # (since increasing the cosine decreases the radius, and we want to maximize min cosine)
        # But we need to be careful about the chain rule through arccos.
        # d(soft_radius)/d(soft_min_cos) = -1/sqrt(1-soft_min_cos^2)
        # d(soft_min_cos)/d(cos_i) = weights[i]
        # d(cos_i)/d(point) ≈ n (for the vertex in question)
        
        # Simplified: move points in direction of -normal * weight for worst facets
        for idx in facet:
            grad_points[idx] -= w * n
    
    # Convert to gradient of soft_radius
    if abs(soft_min_cos) < 1.0:
        chain = -1.0 / np.sqrt(1.0 - soft_min_cos * soft_min_cos)
    else:
        chain = 0.0
    
    grad_points *= chain
    
    return soft_radius, grad_points


def _gradient_descent_smooth(points, deadline, max_iters=2000, base_step=0.03, beta=50.0):
    """Optimize points using gradient descent with smooth covering radius surrogate."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Compute actual radius for tracking
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.02, 0.08)
        else:
            step *= 0.95
        
        if step < 1e-10:
            break
        
        # Compute smooth gradient
        soft_radius, grad = _soft_covering_radius_and_grad(points, beta=beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Move in direction of negative gradient (minimize radius)
        for i in range(N):
            pt = points[i]
            g = grad[i]
            
            # Project gradient to tangent plane
            tangent = g - np.dot(g, pt) * pt
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-12:
                tangent = tangent / tangent_norm
                new_pt = pt - step * tangent
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[i] = new_pt
    
    return points


def _facet_refine(points, deadline, max_iters=1000, base_step=0.02):
    """Refine using direct facet-based moves toward circumcenters."""
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            step = min(step * 1.03, 0.05)
        else:
            step *= 0.93
        
        if step < 1e-10:
            break
        
        # Find worst facets
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
        
        if len(facets) == 0:
            break
        
        # Compute circumcenters and sort by cosine
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
            break
        
        facet_data.sort(key=lambda x: x[0])
        
        # Move points in worst 10 facets
        for cos_val, facet, circ in facet_data[:10]:
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
    
    # Phase 1: Smooth gradient descent from Fibonacci
    best_points = _gradient_descent_smooth(fib_points, deadline, max_iters=1500, base_step=0.03, beta=40.0)
    best_radius = _covering_radius(best_points)
    
    # Phase 2: Facet-based refinement
    if time.time() < deadline - 20.0:
        refined = _facet_refine(best_points, deadline, max_iters=800, base_step=0.015)
        radius = _covering_radius(refined)
        if radius < best_radius:
            best_radius = radius
            best_points = refined
    
    # Phase 3: Random restarts with smooth optimization
    num_restarts = 4
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        # Mix: perturb best or random
        if best_points is not None and restart % 2 == 0:
            perturbation = 0.12 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        max_iters = min(1200, int(remaining * 15))
        optimized = _gradient_descent_smooth(points, deadline, max_iters=max_iters, base_step=0.03, beta=50.0)
        
        # Quick facet refinement
        if time.time() < deadline - 10.0:
            optimized = _facet_refine(optimized, deadline, max_iters=300, base_step=0.01)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        best_points = _facet_refine(best_points, deadline, max_iters=500, base_step=0.005)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
