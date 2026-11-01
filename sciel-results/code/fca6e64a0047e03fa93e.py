"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms

def _objective(points, grid_points):
    # points: (N, 3)
    # grid_points: (G, 3)
    dots = np.dot(grid_points, points.T)
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    min_dists = np.arccos(max_dots)
    
    # Covering radius
    R = np.max(min_dists)
    # Variance of distances
    V = np.var(min_dists)
    
    # Composite objective
    # We want to minimize R primarily, but V helps escape local minima
    return R + 0.05 * V

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)

    # Pre-compute a grid on the sphere for evaluating covering radius
    # Higher resolution grid for better accuracy
    n_lat = 180
    n_lon = 360
    lats = np.linspace(-np.pi/2, np.pi/2, n_lat, endpoint=False) + np.pi/n_lat
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    
    LATS, LONS = np.meshgrid(lats, lons, indexing='ij')
    
    grid_x = np.cos(LATS) * np.cos(LONS)
    grid_y = np.cos(LATS) * np.sin(LONS)
    grid_z = np.sin(LATS)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_points = _normalize(grid_points)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Simulated Annealing Parameters
    T = 0.02  # Initial temperature (in radians)
    T_min = 0.0001
    cooling_rate = 0.998
    step_scale = 0.05
    
    current_obj = _objective(points, grid_points)
    best_points = points.copy()
    best_obj = current_obj
    
    # Track best covering radius separately for final return
    best_radius = np.max(np.arccos(np.clip(np.max(np.dot(grid_points, best_points.T), axis=1), -1.0, 1.0)))
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Generate a candidate move
        idx = np.random.randint(N)
        p = points[idx].copy()
        
        # Generate a random perturbation in tangent space
        rand_vec = np.random.randn(3)
        proj = rand_vec - np.dot(rand_vec, p) * p
        t_norm = np.linalg.norm(proj)
        if t_norm > 1e-9:
            t = proj / t_norm
            new_p = _normalize(p + step_scale * t)
        else:
            continue
            
        # Evaluate new objective
        points[idx] = new_p
        new_obj = _objective(points, grid_points)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_obj = new_obj
            
            # Check if this is the best *covering radius* so far
            # We only care about the final covering radius for the score
            # But we use the composite objective for navigation
            # Let's calculate the actual covering radius for the best tracking
            dots = np.dot(grid_points, points.T)
            dots = np.clip(dots, -1.0, 1.0)
            max_dots = np.max(dots, axis=1)
            min_dists = np.arccos(max_dots)
            curr_radius = np.max(min_dists)
            
            if curr_radius < best_radius:
                best_radius = curr_radius
                best_points = points.copy()
        else:
            # Reject
            points[idx] = p
            
        # Cool down
        T *= cooling_rate
        
    return best_points
# EVOLVE-BLOCK-END
