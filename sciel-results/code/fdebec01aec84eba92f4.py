import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for f in facets:
        a = points[f[0]]
        b = points[f[1]]
        c = points[f[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        m = n / norm_n
        if np.dot(m, a) < 0:
            m = -m
        r = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
        if r > max_r:
            max_r = r
    return max_r


def _compute_facets(points):
    """Compute convex hull facets and their circumradius normals."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    n_facets = len(facets)
    normals = np.zeros((n_facets, 3))
    radii = np.zeros(n_facets)
    valid = np.ones(n_facets, dtype=bool)
    
    for idx, f in enumerate(facets):
        a = points[f[0]]
        b = points[f[1]]
        c = points[f[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            valid[idx] = False
            continue
        m = n / norm_n
        if np.dot(m, a) < 0:
            m = -m
        normals[idx] = m
        radii[idx] = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
    
    return facets, normals, radii


def _soft_max(radii, beta):
    """Smooth approximation of max using log-sum-exp."""
    max_r = np.max(radii)
    if beta < 1e-10:
        return max_r, None
    shifted = radii - max_r
    exp_vals = np.exp(beta * shifted)
    log_sum = np.log(np.sum(exp_vals)) / beta
    return max_r + log_sum, exp_vals / np.sum(exp_vals)


def _smooth_objective_and_grad(x, beta):
    """
    Compute smooth covering radius objective and gradient.
    x: flattened point array (N*3,)
    Returns: (objective_value, gradient)
    """
    points = x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    facets, normals, radii = _compute_facets(points)
    if facets is None:
        return 1.0, np.zeros_like(x)
    
    valid_mask = np.linalg.norm(normals, axis=1) > 1e-10
    if not np.any(valid_mask):
        return 1.0, np.zeros_like(x)
    
    valid_radii = radii[valid_mask]
    valid_normals = normals[valid_mask]
    valid_facets = facets[valid_mask]
    
    obj_val, weights = _soft_max(valid_radii, beta)
    
    if weights is None:
        # beta too small, use hard max gradient
        max_idx = np.argmax(valid_radii)
        f = valid_facets[max_idx]
        m = valid_normals[max_idx]
        grad = np.zeros_like(x)
        for k in f:
            # d(r_i)/d(point_k) where r_i = arccos(m·a_k)
            # dr/dx = -m / sqrt(1 - (m·x)^2) projected tangent
            dot_val = np.dot(m, points[k])
            dot_val = np.clip(dot_val, -1.0 + 1e-15, 1.0 - 1e-15)
            denom = np.sqrt(1.0 - dot_val * dot_val)
            if denom < 1e-15:
                continue
            grad[k] = -m / denom
        # Project to tangent space
        for k in range(N):
            grad[k] = grad[k] - np.dot(grad[k], points[k]) * points[k]
        return obj_val, grad
    
    # Weighted gradient
    grad = np.zeros_like(x)
    for i, (f, m, w) in enumerate(zip(valid_facets, valid_normals, weights)):
        if w < 1e-15:
            continue
        for k in f:
            dot_val = np.dot(m, points[k])
            dot_val = np.clip(dot_val, -1.0 + 1e-15, 1.0 - 1e-15)
            denom = np.sqrt(1.0 - dot_val * dot_val)
            if denom < 1e-15:
                continue
            grad[k] += w * (-m / denom)
    
    # Project to tangent space
    for k in range(N):
        grad[k] = grad[k] - np.dot(grad[k], points[k]) * points[k]
    
    return obj_val, grad


def _lbfgs_refine(points, beta, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Refine points using L-BFGS on the smooth surrogate."""
    x0 = points.flatten()
    
    def fun_and_grad(x):
        val, grad = _smooth_objective_and_grad(x, beta)
        return val, grad
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        final_points = res.x.reshape(N, 3)
        final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
        return final_points
    except Exception:
        return points


def _fibonacci_points(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(n=N, jitter=0.1, seed=42):
    pts = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    pts += rng.normal(0, jitter, pts.shape)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n=N):
    """Generate points based on icosahedron vertices and subdivisions."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Start with icosahedron, then add more points via golden ratio spacing
    pts = [verts]
    # Add Fibonacci points for the rest
    fib = _fibonacci_points(n)
    return fib


def _random_points(n=N, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _refine_multi_stage(points, time_budget):
    """Multi-stage beta annealing refinement."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Beta stages: start smooth, end sharp
    beta_schedule = [10, 50, 150, 300, 500]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget * 0.8:
            break
        
        # Refine with current beta
        refined = _lbfgs_refine(best_points, beta, max_iter=150, ftol=1e-12, gtol=1e-10)
        
        # Evaluate true covering radius
        true_radius = _covering_radius(refined)
        if true_radius < best_radius:
            best_radius = true_radius
            best_points = refined.copy()
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_fibonacci_points())
    
    # Jittered Fibonacci with different seeds
    for seed in [1, 2, 3]:
        initial_configs.append(_jittered_fibonacci_points(jitter=0.05, seed=seed))
    
    # Random starts
    for seed in [42, 123, 456]:
        initial_configs.append(_random_points(seed=seed))
    
    best_points = initial_configs[0].copy()
    best_radius = _covering_radius(best_points)
    
    # Refine each initial configuration
    for i, config in enumerate(initial_configs):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 15.0:
            break
        
        # Allocate time per config
        time_per_config = min(remaining / max(1, len(initial_configs) - i), 25.0)
        
        refined, radius = _refine_multi_stage(config, time_per_config)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined.copy()
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if time_limit - elapsed > 10.0:
        # Very high beta for final precision
        final = _lbfgs_refine(best_points, 1000, max_iter=300, ftol=1e-14, gtol=1e-12)
        final_radius = _covering_radius(final)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final
    
    # Final normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # Safety check
    if time.time() - start_time > time_limit:
        pass  # Shouldn't happen, but just in case
    
    return best_points
