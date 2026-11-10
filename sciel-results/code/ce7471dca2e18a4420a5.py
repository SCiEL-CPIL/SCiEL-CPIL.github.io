import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback for degenerate cases
        
    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        # Normal to the plane of the triangle
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        # Distance from origin to the plane
        d = np.abs(np.dot(normal, p0)) / norm_normal
        
        # Clip d to [0, 1] to avoid numerical errors in arccos
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _repulsive_forces(points, k=10, power=2.0):
    """
    Compute repulsive forces on each point.
    Each point is repelled by all others with strength ~ 1/d^power.
    """
    forces = np.zeros_like(points)
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-14:
                continue
            dist = np.sqrt(dist_sq)
            # Force magnitude ~ 1/dist^power, direction = diff/dist
            force_mag = k / (dist ** power)
            forces[i] += force_mag * diff / dist
    return forces

def _relax_with_forces(points, steps=3000, lr=0.01, k=5.0, power=2.0):
    """
    Relax points using repulsive forces with adaptive learning rate.
    """
    points = points.copy()
    n = len(points)
    
    for step in range(steps):
        forces = _repulsive_forces(points, k=k, power=power)
        
        # Project forces onto tangent plane (remove radial component)
        for i in range(n):
            radial = np.dot(points[i], forces[i])
            forces[i] -= radial * points[i]
        
        # Adaptive learning rate: decrease over time
        current_lr = lr / (1.0 + 0.01 * step)
        
        points += current_lr * forces
        points = _normalize(points)
        
    return points

def _hill_climb(points, max_iter=500, step_size=0.05):
    """
    Simple hill climbing: perturb each point slightly and keep if it improves.
    """
    points = points.copy()
    current_radius = _covering_radius(points)
    n = len(points)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            # Try small random perturbations
            for _ in range(5):
                # Random direction on tangent plane
                rand_dir = np.random.randn(3)
                radial = np.dot(rand_dir, points[i])
                rand_dir -= radial * points[i]
                rand_norm = np.linalg.norm(rand_dir)
                if rand_norm < 1e-10:
                    continue
                rand_dir /= rand_norm
                
                # Try several step sizes
                for ss in [step_size, step_size/2, step_size/4]:
                    new_pt = points[i] + ss * rand_dir
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    
                    old_pt = points[i].copy()
                    points[i] = new_pt
                    
                    new_radius = _covering_radius(points)
                    
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                    else:
                        points[i] = old_pt
                        break  # If this step size didn't help, try smaller
        
        if not improved:
            break
            
    return points, current_radius

def _fibonacci_spiral(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _random_points(n):
    """Generate random points on sphere."""
    v = np.random.randn(n, 3)
    return _normalize(v)

def _spherical_to_cartesian(params):
    """Convert spherical coordinates (theta, phi) to Cartesian."""
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    params = np.zeros(2 * len(points))
    params[0::2] = theta
    params[1::2] = phi
    return params

def _local_optimize(points, deadline, max_iters=200):
    """
    Local optimization using Nelder-Mead on spherical coordinates.
    """
    params0 = _cartesian_to_spherical(points)
    
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = _spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-7, 'fatol': 1e-9})
        if time.time() > deadline:
            return points
        if result.fun < 1e5:  # Valid result
            return _spherical_to_cartesian(result.x)
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = np.pi
    
    def _try_configuration(points, label=""):
        nonlocal best_points, best_radius
        points = _normalize(points)
        
        # Force-based relaxation
        relaxed = _relax_with_forces(points, steps=2000, lr=0.01, k=5.0, power=2.0)
        rad = _covering_radius(relaxed)
        
        if rad < best_radius:
            best_radius = rad
            best_points = relaxed.copy()
        
        # Hill climbing
        if time.time() < deadline - 30:
            hc_points, hc_rad = _hill_climb(relaxed, max_iter=200, step_size=0.03)
            if hc_rad < best_radius:
                best_radius = hc_rad
                best_points = hc_points.copy()
        
        # Nelder-Mead polish
        if time.time() < deadline - 15:
            opt_points = _local_optimize(best_points, deadline, max_iters=150)
            opt_rad = _covering_radius(opt_points)
            if opt_rad < best_radius:
                best_radius = opt_rad
                best_points = opt_points.copy()
        
        return best_points, best_radius
    
    # Try multiple diverse starting configurations
    configs = []
    
    # 1. Fibonacci spiral
    configs.append(_fibonacci_spiral(N))
    
    # 2. Random configurations
    for _ in range(3):
        configs.append(_random_points(N))
    
    # 3. Perturbed Fibonacci
    fib = _fibonacci_spiral(N)
    for _ in range(2):
        noise = np.random.randn(N, 3) * 0.1
        configs.append(_normalize(fib + noise))
    
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 20:
            break
        _try_configuration(cfg, label=f"config_{idx}")
    
    # Final polish if we have time
    if best_points is not None and time.time() < deadline - 5:
        # One more round of hill climbing + Nelder-Mead
        hc_points, hc_rad = _hill_climb(best_points, max_iter=100, step_size=0.02)
        if hc_rad < best_radius:
            best_radius = hc_rad
            best_points = hc_points.copy()
        
        if time.time() < deadline - 3:
            opt_points = _local_optimize(best_points, deadline, max_iters=100)
            opt_rad = _covering_radius(opt_points)
            if opt_rad < best_radius:
                best_radius = opt_rad
                best_points = opt_points.copy()
    
    if best_points is None:
        # Fallback: return Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
