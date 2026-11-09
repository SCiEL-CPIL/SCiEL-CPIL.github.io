"""Optimized covering of the unit sphere with 130 points."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 130


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            
            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)
            
            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                R = max(a, b, c) / 2
            
            if R > max_circumradius:
                max_circumradius = R
        
        return np.degrees(max_circumradius)
    
    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        
        min_dists = np.inf
        for i in range(0, n_samples, 2000):
            chunk = samples[i:i + 2000]
            d = 1 - np.einsum('ij,kj->ik', chunk, points)
            d = np.clip(d, -1, 1)
            min_d = np.min(d, axis=1)
            min_dists = np.minimum(min_dists, min_d)
        
        max_min_dist = np.max(min_dists)
        return np.degrees(np.arccos(np.clip(max_min_dist, -1, 1)))


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def _objective_and_grad(x, beta, samples):
    """
    Objective: soft-max surrogate of covering radius.
    x: flat array of shape (N*3,)
    Returns (objective_value, gradient)
    """
    pts = x.reshape(-1, 3)
    pts = _project_to_sphere(pts)
    n = len(pts)
    
    # cos_d[i, j] = dot(samples[i], pts[j])
    cos_d = np.clip(np.einsum('ij,kj->ik', samples, pts), -1, 1)
    
    # min angle for each sample = arccos(max_j cos_d[i,j])
    max_cos = np.max(cos_d, axis=1)
    min_angles = np.arccos(max_cos)
    
    # Soft-max of min_angles
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    weights = exp_vals / np.sum(exp_vals)
    
    obj = np.sum(weights * min_angles)
    
    # Gradient: d(obj)/d(x)
    # obj = sum_i w_i * arccos(max_j cos_d[i,j])
    # For each sample i, the nearest point j* = argmax_j cos_d[i,j]
    # d/dp_j* arccos(cos_d[i,j*]) = -samples[i] / sqrt(1 - cos_d[i,j*]^2)
    # Then project to tangent space at p_j*
    
    nearest_idx = np.argmax(cos_d, axis=1)
    
    grad_pts = np.zeros_like(pts)
    for i in range(len(samples)):
        j = nearest_idx[i]
        cos_val = cos_d[i, j]
        sin_val = np.sqrt(np.clip(1.0 - cos_val ** 2, 1e-12, 1.0))
        # d arccos(cos)/d p_j = -samples[i] / sin_val
        g = -samples[i] / sin_val
        # Project to tangent space at pts[j]
        g -= np.dot(g, pts[j]) * pts[j]
        grad_pts[j] += weights[i] * g
    
    # Chain rule: d(pts)/d(x) where pts = x/|x|
    # For each point k: d(pts[k])/d(x[k]) = I/|x_k| - pts[k] pts[k]^T / |x_k|
    # = (I - pts[k] pts[k]^T) / |x_k|
    # But since we project, the effective gradient on x is:
    # grad_x[k] = (I - pts[k] pts[k]^T) @ grad_pts[k] / |x_k|
    # Actually, since we're optimizing on the sphere, we can just use grad_pts
    # projected back. The parameterization x -> x/|x| means:
    # d f/d x = (I - p p^T) d f/d p / |x|
    
    norms_x = np.linalg.norm(x.reshape(-1, 3), axis=1, keepdims=True)
    norms_x = np.maximum(norms_x, 1e-10)
    
    grad_x = np.zeros_like(x)
    for k in range(n):
        p = pts[k]
        # Project grad_pts[k] to tangent space (already done partially, but ensure)
        gk = grad_pts[k]
        gk -= np.dot(gk, p) * p
        grad_x[k * 3:(k + 1) * 3] = gk / norms_x[k, 0]
    
    return obj, grad_x


def _optimize_lbfgs(initial_points, max_time, beta=30.0):
    """Optimize using L-BFGS-B on soft-max surrogate."""
    start_time = time.time()
    
    # Generate fixed samples for this optimization run
    rng = np.random.RandomState(777)
    n_samples = 4000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    x0 = initial_points.flatten().copy()
    
    best_x = x0.copy()
    best_obj = np.inf
    
    # Callback to track best and check time
    def callback(xk):
        nonlocal best_x, best_obj
        if time.time() - start_time > max_time:
            return True  # Stop
        obj, _ = _objective_and_grad(xk, beta, samples)
        if obj < best_obj:
            best_obj = obj
            best_x = xk.copy()
        return False
    
    options = {
        'maxiter': 500,
        'maxfun': 500,
        'gtol': 1e-8,
        'ftol': 1e-10,
        'callback': callback
    }
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options=options
        )
        
        if result.success or hasattr(result, 'x'):
            candidate_x = result.x
            obj, _ = _objective_and_grad(candidate_x, beta, samples)
            if obj < best_obj:
                best_obj = obj
                best_x = candidate_x.copy()
    except Exception:
        pass
    
    # Polish with higher beta
    if time.time() - start_time < max_time - 5:
        beta2 = 60.0
        x0 = best_x.copy()
        
        def callback2(xk):
            nonlocal best_x, best_obj
            if time.time() - start_time > max_time:
                return True
            obj, _ = _objective_and_grad(xk, beta2, samples)
            if obj < best_obj:
                best_obj = obj
                best_x = xk.copy()
            return False
        
        options2 = {
            'maxiter': 300,
            'maxfun': 300,
            'gtol': 1e-9,
            'ftol': 1e-11,
            'callback': callback2
        }
        
        try:
            result2 = minimize(
                _objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options=options2
            )
            
            if result2.success or hasattr(result2, 'x'):
                candidate_x = result2.x
                obj, _ = _objective_and_grad(candidate_x, beta2, samples)
                if obj < best_obj:
                    best_obj = obj
                    best_x = candidate_x.copy()
        except Exception:
            pass
    
    final_points = _project_to_sphere(best_x.reshape(-1, 3))
    radius = _covering_radius(final_points)
    
    return final_points, radius


def _multi_restart_optimize(max_time=100):
    """Try multiple initial configurations and return the best one."""
    start_time = time.time()
    deadline = start_time + max_time
    
    best_points = None
    best_radius = np.inf
    
    # Different initial configurations
    configs = []
    
    # Fibonacci spiral
    configs.append(_fibonacci_sphere(N))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        configs.append(_random_sphere(N, seed=seed))
    
    time_per_config = (deadline - time.time()) / len(configs)
    time_per_config = max(5.0, min(12.0, time_per_config))
    
    for i, init_points in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 3:
            break
        
        # Optimize this configuration
        points, radius = _optimize_lbfgs(init_points, max_time=remaining)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # If we've found a very good solution, stop early
        if best_radius < 11.32:
            break
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
        best_radius = _covering_radius(best_points)
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Multi-restart optimization
    points, radius = _multi_restart_optimize(max_time=105)
    
    # Final normalization
    points = _project_to_sphere(points)
    
    # Verify it's valid
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        # Fallback to Fibonacci if something went wrong
        points = _fibonacci_sphere(N)
    
    return points
