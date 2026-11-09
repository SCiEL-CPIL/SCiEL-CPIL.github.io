"""Direct geometric approach: optimize 130 points on unit sphere to minimize covering radius.
Combined approach: Riemannian gradient descent with arccos-based log-sum-exp surrogate,
multi-scale restarts, and L-BFGS polish with proper tangent-space projection."""
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
    Uses arccos(cos_val) for the actual angular distance to get a more accurate gradient.
    Returns (soft_radius, gradient of soft_radius w.r.t. points).
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return np.pi, np.zeros_like(points)
    
    facets, circumcenters, cos_values = _facet_circumcenters(points)
    if len(facets) == 0:
        return np.pi, np.zeros_like(points)
    
    # Angular distance for each facet: theta_j = arccos(cos_j)
    cos_clipped = np.clip(cos_values, -1.0, 1.0)
    angles = np.arccos(cos_clipped)
    
    # Soft max of angles via log-sum-exp
    max_angle = np.max(angles)
    exps = np.exp(beta * (angles - max_angle))
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    
    soft_radius = max_angle + np.log(sum_exps) / beta
    
    # Gradient computation:
    # d(soft_radius)/d(angle_j) = weights[j]
    # d(angle_j)/d(cos_j) = -1/sqrt(1 - cos_j^2)
    # d(cos_j)/d(points[idx]) ≈ p_j for idx in facet j (fixed normal approximation)
    
    sin_angles = np.sqrt(np.maximum(1e-30, 1.0 - cos_clipped * cos_clipped))
    d_angle_d_cos = -1.0 / sin_angles
    
    grad = np.zeros_like(points)
    for j, facet in enumerate(facets):
        w = weights[j] * d_angle_d_cos[j]
        if abs(w) < 1e-15:
            continue
        p_j = circumcenters[j]
        for idx in facet:
            pt = points[idx]
            tangent = p_j - np.dot(p_j, pt) * pt
            grad[idx] += w * tangent
    
    return soft_radius, grad


def _riemannian_gradient_descent(points, deadline, max_iters=2000, base_step=0.03, 
                                   beta_schedule=None, patience_limit=100):
    """
    Riemannian gradient descent using the soft covering radius surrogate.
    Updates points via exponential map on the sphere: exp_p(v) = p*cos(|v|) + (v/|v|)*sin(|v|)
    """
    points = points.copy()
    best_radius = _covering_radius(points)
    step = base_step
    patience = 0
    prev_soft_r = None
    
    if beta_schedule is None:
        beta_schedule = [50.0, 80.0, 120.0, 200.0]
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        # Adaptive beta: increase as we get closer to convergence
        beta_idx = min(len(beta_schedule) - 1, patience // 25)
        beta = beta_schedule[beta_idx]
        
        soft_r, grad = _soft_covering_radius_and_grad(points, beta=beta)
        
        # Check actual radius for best tracking
        radius = _covering_radius(points)
        if radius < best_radius - 1e-10:
            best_radius = radius
            patience = 0
            step = min(step * 1.03, 0.08)
        else:
            patience += 1
            step *= 0.95
            if patience > patience_limit:
                break
        
        if step < 1e-14:
            break
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Riemannian gradient step: move in direction -grad_tangent
        # Use exponential map for sphere
        for i in range(N):
            v = -step * grad_tangent[i]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-15:
                continue
            # exp_p(v) = p*cos(norm_v) + (v/norm_v)*sin(norm_v)
            cos_v = np.cos(norm_v)
            sin_v = np.sin(norm_v) / norm_v
            points[i] = points[i] * cos_v + v * sin_v
            # Renormalize for numerical stability
            points[i] = points[i] / np.linalg.norm(points[i])
    
    return points


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
            step = min(step * 1.02, 0.10)
        else:
            patience += 1
            step *= 0.92
            if patience > 80:
                break
        
        if step < 1e-13:
            break
        
        worst = _worst_facets(points, k=8)
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


def _lbfgs_smooth_optimize(points, deadline, max_iter=500, beta=100.0):
    """Optimize using L-BFGS-B on the soft covering radius surrogate with arccos gradient."""
    points_flat = points.flatten().copy()
    
    def fun(flat_pts):
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        soft_r, _ = _soft_covering_radius_and_grad(pts, beta=beta)
        return float(soft_r)

    def jac(flat_pts):
        pts = flat_pts.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        _, grad = _soft_covering_radius_and_grad(pts, beta=beta)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return grad_tangent.flatten()
    
    result = minimize(fun, points_flat, jac=jac, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                               'maxls': 20, 'maxcor': 20})
    
    optimized = result.x.reshape(N, 3)
    norms = np.linalg.norm(optimized, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    optimized = optimized / norms
    
    return optimized


def _exact_local_polish(points, deadline, max_iters=200):
    """
    High-precision local polish: iteratively move each point in the worst facet
    toward its circumcenter with very small steps, re-evaluating the hull each time.
    This is a coordinate-descent style approach that converges to a local optimum.
    """
    points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iters):
        if time.time() > deadline - 3.0:
            break
        
        facets, circumcenters, cos_values = _facet_circumcenters(points)
        if len(facets) == 0:
            break
        
        # Process worst facets first
        order = np.argsort(cos_values)
        improved = False
        
        for idx in order[:15]:
            if time.time() > deadline - 3.0:
                break
            facet = facets[idx]
            circ = circumcenters[idx]
            cos_val = cos_values[idx]
            
            # Skip if this facet is already very good
            if cos_val > 0.99:
                continue
            
            for pidx in facet:
                pt = points[pidx]
                # Move toward circumcenter
                direction = circ - pt
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-12:
                    continue
                direction = direction / norm_dir
                tangent = direction - np.dot(direction, pt) * pt
                norm_t = np.linalg.norm(tangent)
                if norm_t < 1e-12:
                    continue
                tangent = tangent / norm_t
                
                # Small step
                step = 0.002
                new_pt = pt + step * tangent
                new_pt = new_pt / np.linalg.norm(new_pt)
                points[pidx] = new_pt
                improved = True
        
        if not improved:
            break
        
        radius = _covering_radius(points)
        if radius < best_radius - 1e-12:
            best_radius = radius
    
    return points


def _polish(points, deadline):
    """Final high-precision polish with multiple phases."""
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Phase 1: Riemannian gradient descent with small steps
    if time.time() < deadline - 15.0:
        polished = _riemannian_gradient_descent(points, deadline - 15.0, max_iters=800, 
                                                 base_step=0.005, patience_limit=60)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 2: L-BFGS on smooth surrogate with high beta
    if time.time() < deadline - 10.0:
        polished = _lbfgs_smooth_optimize(best_points, deadline - 10.0, max_iter=300, beta=150.0)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 3: Exact local polish (coordinate descent on worst facets)
    if time.time() < deadline - 5.0:
        polished = _exact_local_polish(best_points, deadline - 5.0, max_iters=150)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 4: One more L-BFGS pass with even higher beta if time allows
    if time.time() < deadline - 2.0:
        polished = _lbfgs_smooth_optimize(best_points, deadline - 2.0, max_iter=150, beta=250.0)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    # Phase 5: Final exact polish
    if time.time() < deadline - 1.0:
        polished = _exact_local_polish(best_points, deadline - 1.0, max_iters=80)
        r = _covering_radius(polished)
        if r < best_radius:
            best_points = polished
            best_radius = r
    
    return best_points


def _fibonacci_spiral(n, offset=0.5):
    """Generate points via Fibonacci spiral on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.pi
    
    # Start with Fibonacci spiral as baseline
    fib_points = _fibonacci_spiral(N)
    
    # Initial optimization: Riemannian gradient descent
    best_points = _riemannian_gradient_descent(fib_points, deadline, max_iters=1500, base_step=0.04)
    best_radius = _covering_radius(best_points)
    
    # Multiple restarts with varied strategies
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 45.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        # Vary perturbation strategy
        if restart == 0:
            # Small perturbation from best
            perturbation = 0.05 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 1:
            # Moderate perturbation
            perturbation = 0.12 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Full random restart
            points = rng.standard_normal((N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Fibonacci with different offset
            points = _fibonacci_spiral(N, offset=0.3 + 0.1 * restart)
        elif restart == 4:
            # Larger perturbation
            perturbation = 0.25 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 5:
            # Perturb only a subset of points
            points = best_points.copy()
            subset = rng.choice(N, size=N // 2, replace=False)
            perturbation = 0.15 * rng.standard_normal((N, 3))
            points[subset] += perturbation[subset]
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 6:
            # Spherical cap perturbation: shift a cluster
            points = best_points.copy()
            center = rng.standard_normal(3)
            center = center / np.linalg.norm(center)
            mask = (points @ center) > 0.3
            perturbation = 0.1 * rng.standard_normal((N, 3))
            points[mask] += perturbation[mask]
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Very small perturbation for fine-tuning
            perturbation = 0.03 * rng.standard_normal((N, 3))
            points = best_points + perturbation
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize with Riemannian GD
        max_iters = min(1200, int(remaining * 10))
        optimized = _riemannian_gradient_descent(points, deadline, max_iters=max_iters, base_step=0.04)
        
        radius = _covering_radius(optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 18.0:
        best_points = _polish(best_points, deadline)
        best_radius = _covering_radius(best_points)
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_points
    
    # Final normalization and validation
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_points = best_points / norms
    
    # Verify no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                # Separate them
                best_points[j] += 0.01 * rng.standard_normal(3)
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
