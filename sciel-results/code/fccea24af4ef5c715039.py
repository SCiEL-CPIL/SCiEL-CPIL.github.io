"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Fast covering radius computation via convex hull facet circumradii.
    Returns the covering radius in radians, or np.pi if invalid.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Vectorized coincident point check
    dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    simplices = hull.simplices
    max_circumradius = 0.0
    
    # Vectorize the circumcenter computation
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Normal to each facet
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = (norms[:, 0] > 1e-12)
    if not np.all(valid):
        return np.pi
    
    normals = normals[valid] / norms[valid]
    a_v = a[valid]
    b_v = b[valid]
    c_v = c[valid]
    
    # For each facet, compute angular distance from normal to each vertex
    # The circumcenter is either +n or -n; pick the one with smaller max distance
    dot_a = np.clip(normals @ a_v.T, -1, 1)
    dot_b = np.clip(normals @ b_v.T, -1, 1)
    dot_c = np.clip(normals @ c_v.T, -1, 1)
    
    # Angular distances for +n
    da_pos = np.arccos(dot_a)
    db_pos = np.arccos(dot_b)
    dc_pos = np.arccos(dot_c)
    max_pos = np.maximum(np.maximum(da_pos, db_pos), dc_pos)
    
    # Angular distances for -n
    da_neg = np.arccos(-dot_a)
    db_neg = np.arccos(-dot_b)
    dc_neg = np.arccos(-dot_c)
    max_neg = np.maximum(np.maximum(da_neg, db_neg), dc_neg)
    
    # For each facet, the circumradius is min(max_pos, max_neg)
    # but only if the three distances are approximately equal (valid circumcenter)
    # In practice, for points on a sphere, one of ±n will give equal distances
    facet_radii = np.minimum(max_pos, max_neg)
    
    max_circumradius = np.max(facet_radii)
    return max_circumradius

def _smooth_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimization.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Compute all pairwise angular distances
    dots = np.clip(pts @ pts.T, -1, 1)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    # For each point on the sphere (approximated by the vertices), find min distance
    min_dists = np.min(dists, axis=1)
    
    # Log-sum-exp for smooth max
    max_d = np.max(min_dists)
    if max_d < 1e-10:
        return 0.0
    
    shifted = min_dists - max_d
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_d + log_sum

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedron_based(n):
    """
    Generate a more uniform starting configuration based on icosahedron vertices
    plus additional points to reach n points.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n <= 12:
        return ico[:n]
    
    # Add more points using Fibonacci spiral in the gaps
    remaining = n - 12
    extra = _fibonacci_spiral(remaining, offset=np.pi/7)
    pts = np.vstack([ico, extra])
    return pts

def _local_optimize(pts, max_iter=300, step_size=0.05, deadline=None):
    """
    Local optimization with adaptive step size and optional deadline.
    """
    best_pts = pts.copy()
    best_val = _covering_radius_fast(best_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random point
        idx = np.random.randint(len(best_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + current_step * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            current_step *= 1.02
        else:
            current_step *= 0.95
        
        if current_step < 1e-6:
            break
    
    return best_pts, best_val

def _simulated_annealing(pts, max_iter=500, T0=0.3, T_end=0.005, deadline=None):
    """
    Simulated annealing for covering radius minimization.
    """
    current_pts = pts.copy()
    current_val = _covering_radius_fast(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        T *= cooling
        
        # Perturb a random point
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.08 * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-10 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
    
    return best_pts, best_val

def _gradient_refine(pts, max_iter=100, deadline=None):
    """
    Gradient-based refinement using the smooth surrogate.
    Uses projected gradient descent on the sphere.
    """
    current_pts = pts.copy()
    current_val = _smooth_covering_radius(current_pts, beta=80.0)
    best_pts = current_pts.copy()
    best_val = _covering_radius_fast(current_pts)
    
    lr = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient of smooth covering radius
        # Use finite differences for simplicity (or analytic gradient)
        eps = 1e-5
        grad = np.zeros_like(current_pts)
        
        for i in range(len(current_pts)):
            for j in range(3):
                # Projected gradient: move in direction, then project back to sphere
                perturbed = current_pts.copy()
                perturbed[i, j] += eps
                perturbed[i] = perturbed[i] / np.linalg.norm(perturbed[i])
                val_plus = _smooth_covering_radius(perturbed, beta=80.0)
                
                perturbed2 = current_pts.copy()
                perturbed2[i, j] -= eps
                perturbed2[i] = perturbed2[i] / np.linalg.norm(perturbed2[i])
                val_minus = _smooth_covering_radius(perturbed2, beta=80.0)
                
                grad[i, j] = (val_plus - val_minus) / (2 * eps)
        
        # Project gradient onto tangent space
        for i in range(len(current_pts)):
            grad[i] -= np.dot(grad[i], current_pts[i]) * current_pts[i]
        
        # Gradient descent step
        new_pts = current_pts - lr * grad
        new_pts = _to_sphere(new_pts)
        
        new_smooth_val = _smooth_covering_radius(new_pts, beta=80.0)
        
        # Check actual covering radius
        new_val = _covering_radius_fast(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts.copy()
            best_val = new_val
        
        if new_smooth_val < current_val:
            current_pts = new_pts
            current_val = new_smooth_val
            lr *= 1.01
        else:
            lr *= 0.9
        
        if lr < 1e-8:
            break
    
    return best_pts, best_val

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses multiple strategies: Fibonacci, icosahedron-based, random restarts,
    with local optimization, simulated annealing, and gradient refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_pts = None
    best_val = np.pi
    
    # Strategy 1: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    val = _covering_radius_fast(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
    
    # Local optimization from Fibonacci
    opt_pts, opt_val = _local_optimize(pts, max_iter=150, step_size=0.04, deadline=deadline)
    if opt_val < best_val:
        best_val = opt_val
        best_pts = opt_pts.copy()
    
    # Strategy 2: Icosahedron-based start
    pts_ico = _icosahedron_based(N)
    val_ico = _covering_radius_fast(pts_ico)
    if val_ico < best_val:
        best_val = val_ico
        best_pts = pts_ico.copy()
    
    opt_pts_ico, opt_val_ico = _local_optimize(pts_ico, max_iter=150, step_size=0.04, deadline=deadline)
    if opt_val_ico < best_val:
        best_val = opt_val_ico
        best_pts = opt_pts_ico.copy()
    
    # Simulated annealing from best so far
    sa_pts, sa_val = _simulated_annealing(best_pts, max_iter=400, T0=0.25, T_end=0.005, deadline=deadline)
    if sa_val < best_val:
        best_val = sa_val
        best_pts = sa_pts.copy()
    
    # Multiple random restarts
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random start
        pts_r = np.random.randn(N, 3)
        pts_r = pts_r / np.linalg.norm(pts_r, axis=1, keepdims=True)
        
        # Local optimization
        opt_pts_r, opt_val_r = _local_optimize(pts_r, max_iter=80, step_size=0.05, deadline=deadline)
        
        # Simulated annealing
        sa_pts_r, sa_val_r = _simulated_annealing(opt_pts_r, max_iter=250, T0=0.2, T_end=0.005, deadline=deadline)
        
        if sa_val_r < best_val:
            best_val = sa_val_r
            best_pts = sa_pts_r.copy()
    
    # Gradient refinement pass
    if time.time() < deadline - 10.0:
        grad_pts, grad_val = _gradient_refine(best_pts, max_iter=80, deadline=deadline)
        if grad_val < best_val:
            best_val = grad_val
            best_pts = grad_pts.copy()
    
    # Final local polish
    if time.time() < deadline - 5.0:
        final_pts, final_val = _local_optimize(best_pts, max_iter=100, step_size=0.02, deadline=deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
