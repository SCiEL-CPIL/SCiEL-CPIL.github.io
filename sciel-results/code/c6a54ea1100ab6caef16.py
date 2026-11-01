import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius_deg_fast(config, grid_size=20000):
    """
    Fast covering radius estimation using dense grid sampling.
    """
    grid = _fib_sample(grid_size)
    dots = grid @ config.T
    min_dots = np.min(dots, axis=1)
    max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.degrees(np.max(max_angle))

def _covering_radius_deg_hull(config):
    """
    Compute covering radius using convex hull facets (more accurate but slower).
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return _covering_radius_deg_fast(config, 30000)
    
    max_angle = 0.0
    for simplex in hull.simplices:
        pts = config[simplex]
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            continue
        
        c = c / c_norm
        
        # Check if circumcenter is valid
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        if np.max(angles) - np.min(angles) > 1e-6:
            # Worst point is a vertex
            max_in_facet = np.max(angles)
        else:
            max_in_facet = angles[0]
        
        # Check nearest config point to circumcenter
        dots_all = np.clip(config @ c, -1.0, 1.0)
        min_angle_to_c = np.arccos(np.max(dots_all))
        
        if min_angle_to_c > max_angle:
            max_angle = min_angle_to_c
    
    return np.degrees(max_angle)

def _local_optimize(config, deadline, verbose=False):
    """
    Local optimization using simultaneous moves of multiple points.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_fast(config, 10000)
    current_score = best_score
    
    step_size = 0.02
    no_improve = 0
    
    while time.time() < deadline - 2.0:
        # Find worst region using fast method
        grid = _fib_sample(10000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_idx = np.argmax(min_dots)
        worst_point = grid[max_idx]
        worst_dot = min_dots[max_idx]
        worst_angle = np.arccos(np.clip(worst_dot, -1.0, 1.0))
        
        # Find the 3 nearest config points to the worst point
        all_dots = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-all_dots)
        nearest_3 = sorted_idx[:3]
        
        # Try moving each of the 3 nearest points toward the worst point
        improved = False
        for idx in nearest_3:
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            new_pt = config[idx] + step_size * v_proj / v_norm
            new_pt = _normalize(new_pt)
            old_pt = config[idx].copy()
            config[idx] = new_pt
            
            new_score = _covering_radius_deg_fast(config, 8000)
            if new_score < current_score - 1e-8:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_config = config.copy()
                no_improve = 0
                break
            else:
                config[idx] = old_pt
        
        if not improved:
            no_improve += 1
            step_size *= 0.85
            if step_size < 0.001 or no_improve > 30:
                break
    
    # Final refinement with hull-based method
    if time.time() < deadline - 1.0:
        final_score = _covering_radius_deg_hull(best_config)
        if final_score < best_score:
            best_score = final_score
    
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run local optimization
    config = _local_optimize(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
