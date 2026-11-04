"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _normalize_points(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _get_circumradius(p1, p2, p3):
    """
    Calculate the circumradius of a spherical triangle defined by 3 unit vectors.
    """
    n = np.cross(p1 - p2, p2 - p3)
    c = _normalize(n)
    # Ensure c is on the same side as the vertices (dot product positive)
    if np.dot(c, p1) < 0:
        c = -c
    cos_r = np.dot(c, p1)
    # Clamp for numerical stability
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c

def _compute_covering_radius(points):
    """
    Compute the covering radius of the configuration in radians.
    Returns max_r, worst_facet_indices, worst_center.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        worst_facet = None
        worst_center = None
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            p3 = points[simplex[2]]
            r, c = _get_circumradius(p1, p2, p3)
            if r > max_r:
                max_r = r
                worst_facet = simplex
                worst_center = c
        return max_r, worst_facet, worst_center
    except Exception:
        # If convex hull fails, return a large value
        return np.pi, None, None

def _compute_radii_vectorized(pts):
    """
    Compute circumradii for all facets of the convex hull.
    Returns (radii, simplices, centers) or None if hull fails.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return None
        
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Normal to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Handle degenerate triangles
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return None
        
    n = n / norm_n
    
    # Ensure normals point towards the interior of the sphere relative to the triangle
    dots = np.sum(n * p1, axis=1)
    n[dots < 0] *= -1
    
    # Calculate circumradius
    cos_r = np.sum(n * p1, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    return radii, simplices, n

def _smooth_facet_surrogate(params, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    result = _compute_radii_vectorized(pts)
    if result is None:
        return np.pi
    
    radii, simplices, centers = result
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta

def _smooth_facet_surrogate_grad(params, beta=50.0):
    """
    Compute value and gradient of the smooth facet surrogate.
    Uses finite differences for the gradient (expensive but reliable).
    """
    val = _smooth_facet_surrogate(params, beta)
    
    # Use finite differences for gradient
    grad = np.zeros_like(params)
    h = 1e-6
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        val_plus = _smooth_facet_surrogate(params, beta)
        params[i] = orig - h
        val_minus = _smooth_facet_surrogate(params, beta)
        params[i] = orig
        grad[i] = (val_plus - val_minus) / (2 * h)
    
    return val, grad

def _repulsive_energy(params, s=20.0):
    """
    Riesz s-energy with high exponent to maximize minimum pairwise separation.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    # Compute all pairwise distances
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)
    
    dist_sq = 2.0 - 2.0 * dots
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    dist_sq = np.maximum(dist_sq, 1e-10)
    
    # Riesz s-energy: sum (1/d^s)
    return np.sum(dist_sq ** (-s / 2.0))

def _repulsive_energy_grad(params, s=20.0):
    """
    Compute value and gradient of Riesz s-energy.
    """
    pts = params.reshape(N, 3)
    pts = _normalize_points(pts)
    
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)
    
    dist_sq = 2.0 - 2.0 * dots
    
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = dist_sq[mask]
    dist_sq_masked = np.maximum(dist_sq_masked, 1e-10)
    
    # Energy
    energy = np.sum(dist_sq_masked ** (-s / 2.0))
    
    # Gradient via finite differences
    grad = np.zeros_like(params)
    h = 1e-6
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + h
        e_plus = _repulsive_energy(params, s)
        params[i] = orig - h
        e_minus = _repulsive_energy(params, s)
        params[i] = orig
        grad[i] = (e_plus - e_minus) / (2 * h)
    
    return energy, grad

def _gradient_descent_phase(points, time_limit=20.0, beta=50.0, use_repulsive=False, s=20.0, use_grad=True):
    """
    Perform gradient descent using L-BFGS-B on the smooth surrogate.
    If use_repulsive is True, uses Riesz s-energy instead.
    """
    start_time = time.time()
    
    if use_repulsive:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _repulsive_energy_grad(params, s=s)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _repulsive_energy(params, s=s)
            gradient = None
    else:
        if use_grad:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _smooth_facet_surrogate(params, beta=beta)
            def gradient(params):
                if time.time() - start_time > time_limit:
                    return np.zeros_like(params)
                _, g = _smooth_facet_surrogate_grad(params, beta=beta)
                return g
        else:
            def objective(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return _smooth_facet_surrogate(params, beta=beta)
            gradient = None
    
    x0 = points.flatten()
    
    try:
        if gradient is not None:
            result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300})
        else:
            result = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300})
        best_points = _normalize_points(result.x.reshape(N, 3))
        return best_points
    except Exception:
        return points

def _facet_shrink_polish(points, time_limit=15.0):
    """
    Geometric relaxation: move worst facet vertices toward circumcenter.
    """
    start_time = time.time()
    step = 0.03
    
    for it in range(5000):
        if time.time() > start_time + time_limit:
            break
            
        current_radius, worst_facet, worst_center = _compute_covering_radius(points)
        
        if worst_facet is None:
            break
            
        v0 = points[worst_facet[0]]
        v1 = points[worst_facet[1]]
        v2 = points[worst_facet[2]]
        
        def move_vertex(v, c, step):
            grad = c - np.dot(c, v) * v
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                return v
            grad = grad / grad_norm
            v_new = v + step * grad
            return _normalize(v_new)
        
        points[worst_facet[0]] = move_vertex(v0, worst_center, step)
        points[worst_facet[1]] = move_vertex(v1, worst_center, step)
        points[worst_facet[2]] = move_vertex(v2, worst_center, step)
        
        # Decay step size
        if it % 50 == 0:
            step *= 0.97
            if step < 0.0005:
                step = 0.0005

    return points

def _multi_restart_optimize(points, deadline, num_restarts=3):
    """
    Try multiple perturbations and keep the best result.
    """
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Create a perturbed version
        perturbation_scale = 0.05 * (restart + 1)
        noise = np.random.randn(N, 3) * perturbation_scale
        perturbed = _normalize_points(points + noise)
        
        # Optimize from perturbed start
        optimized = _gradient_descent_phase(perturbed, time_limit=15.0, beta=30.0, use_grad=False)
        
        # Polish
        polished = _facet_shrink_polish(optimized, time_limit=8.0)
        
        radius, _, _ = _compute_covering_radius(polished)
        if radius < best_radius:
            best_radius = radius
            best_points = polished.copy()
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius(points)
    
    # Phase 1: Coarse optimization with low beta
    if time.time() < deadline - 80.0:
        points = _gradient_descent_phase(points, time_limit=25.0, beta=8.0, use_grad=False)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Medium beta
    if time.time() < deadline - 50.0:
        points = _gradient_descent_phase(points, time_limit=20.0, beta=30.0, use_grad=False)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 3: High beta polishing
    if time.time() < deadline - 25.0:
        points = _gradient_descent_phase(points, time_limit=15.0, beta=100.0, use_grad=False)
        current_radius, _, _ = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 4: Repulsive energy refinement
    if time.time() < deadline - 15.0:
        repulsive_points = best_points.copy()
        repulsive_points = _gradient_descent_phase(repulsive_points, time_limit=10.0, use_repulsive=True, s=30.0, use_grad=False)
        repulsive_radius, _, _ = _compute_covering_radius(repulsive_points)
        if repulsive_radius < best_radius:
            best_radius = repulsive_radius
            best_points = repulsive_points.copy()
            # Follow up with facet polish
            if time.time() < deadline - 5.0:
                polished = _facet_shrink_polish(best_points, time_limit=5.0)
                pol_radius, _, _ = _compute_covering_radius(polished)
                if pol_radius < best_radius:
                    best_radius = pol_radius
                    best_points = polished.copy()
    
    # Phase 5: Multi-restart exploration
    if time.time() < deadline - 20.0:
        best_points, best_radius = _multi_restart_optimize(best_points, deadline, num_restarts=2)
    
    # Phase 6: Final facet shrink polish
    if time.time() < deadline - 3.0:
        polished = _facet_shrink_polish(best_points, time_limit=10.0)
        pol_radius, _, _ = _compute_covering_radius(polished)
        if pol_radius < best_radius:
            best_radius = pol_radius
            best_points = polished.copy()
    
    # Final high-beta gradient polish if time allows
    if time.time() < deadline - 2.0:
        final_points = _gradient_descent_phase(best_points, time_limit=3.0, beta=200.0, use_grad=False)
        final_radius, _, _ = _compute_covering_radius(final_points)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()

    # Final check
    if np.all(np.isfinite(best_points)) and np.all(np.linalg.norm(best_points, axis=1) > 1e-9):
        return best_points
    else:
        return points
# EVOLVE-BLOCK-END
