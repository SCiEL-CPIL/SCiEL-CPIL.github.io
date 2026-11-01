import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

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

def _covering_radius_deg(config):
    """
    Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum over all hull facets of the angular
    distance from the facet's circumcenter (if valid) or the worst vertex
    to the nearest configuration point.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback: dense sampling
        grid = _fib_sample(30000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    max_angle = 0.0
    for simplex in hull.simplices:
        pts = config[simplex]  # (3, 3)
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        
        if c_norm < 1e-12:
            # Degenerate facet
            # The worst point is likely a vertex
            for p in pts:
                dots = np.clip(config @ p, -1.0, 1.0)
                ang = np.arccos(np.max(dots))
                if ang > max_angle:
                    max_angle = ang
            continue
        
        c = c / c_norm
        
        # Check if circumcenter is valid (equidistant from all 3 vertices)
        dots = np.clip(pts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        
        if np.max(angles) - np.min(angles) > 1e-5:
            # Circumcenter is outside the triangle; worst point is a vertex
            for p in pts:
                dots_all = np.clip(config @ p, -1.0, 1.0)
                ang = np.arccos(np.max(dots_all))
                if ang > max_angle:
                    max_angle = ang
        else:
            # Circumcenter is valid and inside the triangle
            # The worst point in this facet is the circumcenter
            dots_all = np.clip(config @ c, -1.0, 1.0)
            ang = np.arccos(np.max(dots_all))
            if ang > max_angle:
                max_angle = ang
    
    return np.degrees(max_angle)

def _lloyd_step(config, grid):
    """One step of Lloyd's algorithm using a fixed grid."""
    dots = grid @ config.T
    max_idx = np.argmax(dots, axis=1)
    
    new_config = np.zeros_like(config)
    for i in range(N):
        mask = (max_idx == i)
        count = np.sum(mask)
        if count > 0:
            center = np.mean(grid[mask], axis=0)
            new_config[i] = _normalize(center)
        else:
            new_config[i] = _normalize(np.random.randn(3))
    return new_config

def _refine(config, deadline):
    """
    Refine the configuration using gradient-based optimization on the
    covering radius computed from hull facets.
    """
    start_time = time.time()
    config = config.copy()
    
    # Use a moderate grid for Lloyd's
    grid = _fib_sample(8000)
    
    # Run Lloyd's algorithm for a few iterations
    for _ in range(30):
        if time.time() > deadline - 60:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
    
    # Now do local refinement
    # We'll use a local search approach: find the worst facet, move points
    
    current_score = _covering_radius_deg(config)
    best_config = config.copy()
    best_score = current_score
    
    step_size = 0.03
    stagnation = 0
    
    while time.time() < deadline - 10:
        # Find the worst point on the sphere
        # We'll sample the sphere densely and find the point with max angular distance
        # But that's expensive. Instead, use hull facets.
        
        try:
            hull = ConvexHull(config)
        except Exception:
            break
        
        worst_angle = 0.0
        worst_point = None
        worst_facet = None
        
        for simplex in hull.simplices:
            pts = config[simplex]
            
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            c = np.cross(v1, v2)
            c_norm = np.linalg.norm(c)
            
            if c_norm < 1e-12:
                continue
            
            c = c / c_norm
            
            dots = np.clip(pts @ c, -1.0, 1.0)
            angles = np.arccos(dots)
            
            if np.max(angles) - np.min(angles) > 1e-5:
                # Worst point is a vertex
                for idx in simplex:
                    p = config[idx]
                    dots_all = np.clip(config @ p, -1.0, 1.0)
                    ang = np.arccos(np.max(dots_all))
                    if ang > worst_angle:
                        worst_angle = ang
                        worst_point = p
                        worst_facet = simplex
            else:
                # Worst point is the circumcenter
                dots_all = np.clip(config @ c, -1.0, 1.0)
                ang = np.arccos(np.max(dots_all))
                if ang > worst_angle:
                    worst_angle = ang
                    worst_point = c
                    worst_facet = simplex
        
        if worst_point is None:
            break
        
        # Find the nearest configuration point to the worst point
        dots = np.clip(worst_point @ config.T, -1.0, 1.0)
        sorted_idx = np.argsort(-dots)
        
        moved = False
        for idx in sorted_idx:
            v = worst_point - config[idx]
            v_proj = v - np.dot(v, config[idx]) * config[idx]
            v_norm = np.linalg.norm(v_proj)
            
            if v_norm < 1e-10:
                continue
            
            # Try moving this point towards the worst point
            new_pt = config[idx] + step_size * v_proj
            new_pt = _normalize(new_pt)
            
            old_pt = config[idx].copy()
            config[idx] = new_pt
            
            new_score = _covering_radius_deg(config)
            
            if new_score < current_score - 1e-10:
                current_score = new_score
                stagnation = 0
                if new_score < best_score:
                    best_score = new_score
                    best_config = config.copy()
                moved = True
            else:
                config[idx] = old_pt
        
        if not moved:
            stagnation += 1
            step_size *= 0.9
            if step_size < 0.0005 or stagnation > 30:
                # Try a random perturbation
                idx = np.random.randint(N)
                config[idx] = _normalize(config[idx] + 0.01 * np.random.randn(3))
                new_score = _covering_radius_deg(config)
                if new_score < current_score:
                    current_score = new_score
                    stagnation = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_config = config.copy()
                else:
                    config[idx] = _normalize(best_config[idx])
                    if stagnation > 50:
                        break
    
    return best_config

def _optimize_from_start(config, deadline):
    """Full optimization pipeline from a starting configuration."""
    start_time = time.time()
    
    # Lloyd's algorithm with a grid
    grid = _fib_sample(10000)
    
    for _ in range(50):
        if time.time() > deadline - 80:
            break
        new_config = _lloyd_step(config, grid)
        if np.allclose(config, new_config, atol=1e-9):
            break
        config = new_config
    
    # Refine
    config = _refine(config, deadline)
    
    return config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Optimize
    config = _optimize_from_start(config, deadline)
    
    # Final normalization and validation
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
    
    return config
