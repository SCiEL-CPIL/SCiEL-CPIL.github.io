import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 120.0  # Safety margin from 150s

def _covering_radius_fast(points, grid_size=60):
    """
    Fast estimate of covering radius using a grid on the sphere.
    For each grid point, find the nearest configuration point and compute angular distance.
    Returns the maximum angular distance in radians.
    """
    # Generate grid points on sphere using Fibonacci spiral for better coverage
    n_grid = grid_size * grid_size
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute all pairwise dot products
    # grid_pts: (n_grid, 3), points: (N, 3)
    dots = grid_pts @ points.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Nearest point for each grid point
    max_dots = np.max(dots, axis=1)
    min_angles = np.arccos(max_dots)
    
    return np.max(min_angles)

def _covering_radius_smooth(points, beta=50.0, grid_size=60):
    """
    Smooth max of angular distances using log-sum-exp over a grid.
    Returns (soft_max_value, max_value).
    """
    n_grid = grid_size * grid_size
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid_pts @ points.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    
    max_angle = np.max(angles)
    
    # Soft max via log-sum-exp
    shifted = beta * angles
    shifted -= np.max(shifted)
    soft_max = np.max(angles) + np.log(np.sum(np.exp(shifted))) / beta
    
    return soft_max, max_angle

def _compute_exact_covering_radius(points):
    """
    Compute exact covering radius using convex hull circumradii.
    Returns covering radius in radians.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.pi / 2
    
    max_theta = 0.0
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            continue
            
        c = n / norm_n
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0, 1.0)
        
        if dot < 0:
            dot = -dot
            
        theta = np.arccos(dot)
        max_theta = max(max_theta, theta)
    
    return max_theta

def _objective_fast(x, beta=50.0, grid_size=60):
    """
    Fast objective: smooth max of angular distances + sphere constraint penalty.
    """
    points = x.reshape(N, 3)
    
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 50.0
    
    soft_max, _ = _covering_radius_smooth(points, beta, grid_size)
    
    return soft_max + sphere_penalty

def _local_optimize(start_points, beta_start=20.0, beta_end=100.0, total_iters=300, grid_size=60):
    """
    Run local optimization with adaptive beta annealing using fast objective.
    """
    x0 = start_points.flatten()
    current = x0.copy()
    
    n_phases = 4
    betas = np.linspace(beta_start, beta_end, n_phases)
    iters_per_phase = max(10, total_iters // n_phases)
    
    for i, beta in enumerate(betas):
        try:
            res = scipy.optimize.minimize(
                _objective_fast,
                current,
                args=(beta, grid_size),
                method='L-BFGS-B',
                options={'maxiter': iters_per_phase, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current = res.x
        except Exception:
            pass
    
    x_opt = current.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    # Evaluate with exact covering radius
    score = _compute_exact_covering_radius(x_opt)
    
    return x_opt, score

def _polish(points, maxiter=300, grid_size=80):
    """
    Final high-precision polish pass with higher resolution grid.
    """
    x0 = points.flatten()
    
    try:
        res = scipy.optimize.minimize(
            _objective_fast,
            x0,
            args=(150.0, grid_size),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-10}
        )
    except Exception:
        return points
    
    x_opt = res.x.reshape(N, 3)
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt

def _fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate random points on sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _jittered_fibonacci(n, jitter_scale, rng):
    """Generate jittered Fibonacci sphere points."""
    base = _fibonacci_sphere(n)
    jitter = jitter_scale * rng.standard_normal((n, 3))
    pts = base + jitter
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def construct():
    start_time = time.time()
    best_points = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    for _ in range(6):
        starts.append(_random_sphere(N, rng))
    
    fib = _fibonacci_sphere(N)
    for scale in [0.05, 0.1, 0.15]:
        starts.append(_jittered_fibonacci(N, scale, rng))
    
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        # Use fewer iterations for later starts if time is tight
        remaining = TIME_LIMIT - 20 - elapsed
        if remaining < 10:
            break
            
        points, score = _local_optimize(start, beta_start=20.0, beta_end=100.0, total_iters=250, grid_size=60)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15 and best_points is not None:
        best_points = _polish(best_points, maxiter=200, grid_size=80)
        best_score = _compute_exact_covering_radius(best_points)
    
    # Final polish with even higher precision if time allows
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 8 and best_points is not None:
        best_points = _polish(best_points, maxiter=100, grid_size=100)
        best_score = _compute_exact_covering_radius(best_points)
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
