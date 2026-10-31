"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _precompute_indices():
    """Precompute all (i,j,k) index triples for N points."""
    idxs = list(combinations(range(N), 3))
    return np.array(idxs)

_IDX = _precompute_indices()
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

def _min_triangle_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Vectorized for speed.
    points: (N, 2) array
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    cross = v1x * v2y - v1y * v2x
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle_fast(points):
    """Clamp all points to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    h = np.sqrt(3) / 2
    eps = 1e-9
    
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    y = np.clip(y, eps, h - eps)
    
    x_min_left = y / np.sqrt(3)
    x_max_right = 1.0 - y / np.sqrt(3)
    
    x = np.clip(x, x_min_left + eps, x_max_right - eps)
    
    points[:, 0] = x
    points[:, 1] = y
    return points

def _local_optimize(points, time_limit=5.0):
    """
    Custom local optimizer: iteratively move the point that is in the smallest triangle
    to maximize the minimum area. Uses a simple coordinate descent / hill climbing.
    """
    points = np.array(points, dtype=float)
    _clamp_to_triangle_fast(points)
    
    best_points = points.copy()
    best_score = _min_triangle_area_vec(points)
    start_time = time.time()
    
    # Find which triple gives the minimum area
    def _get_min_triple(pts):
        pi = pts[_I]
        pj = pts[_J]
        pk = pts[_K]
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        cross = v1x * v2y - v1y * v2y
        areas = 0.5 * np.abs(cross)
        min_idx = np.argmin(areas)
        return min_idx, areas[min_idx]
    
    # Simple hill climbing: try moving each point in small steps
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    rng = np.random.default_rng(12345)
    
    for step in step_sizes:
        if time.time() - start_time > time_limit:
            break
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            improved = False
            for idx in range(N):
                if time.time() - start_time > time_limit:
                    break
                # Try 8 directions + random
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    dx = step * np.cos(angle)
                    dy = step * np.sin(angle)
                    new_pt = points[idx].copy()
                    new_pt[0] += dx
                    new_pt[1] += dy
                    _clamp_to_triangle_fast(new_pt.reshape(1,2))
                    new_pt = new_pt.reshape(1,2)[0]
                    
                    old_pt = points[idx].copy()
                    points[idx] = new_pt
                    score = _min_triangle_area_vec(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        improved = True
                    else:
                        points[idx] = old_pt
                
                # Also try a random direction
                angle = rng.random() * 2 * np.pi
                dx = step * np.cos(angle)
                dy = step * np.sin(angle)
                new_pt = points[idx].copy()
                new_pt[0] += dx
                new_pt[1] += dy
                _clamp_to_triangle_fast(new_pt.reshape(1,2))
                new_pt = new_pt.reshape(1,2)[0]
                
                old_pt = points[idx].copy()
                points[idx] = new_pt
                score = _min_triangle_area_vec(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    improved = True
                else:
                    points[idx] = old_pt
    
    return best_points, best_score

def _simulated_annealing(start_points, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    points = np.array(start_points, dtype=float)
    _clamp_to_triangle_fast(points)
    current_score = _min_triangle_area_vec(points)
    
    best_points = points.copy()
    best_score = current_score
    
    # Parameters
    T = 0.005
    T_min = 1e-7
    alpha = 0.9997
    step_size = 0.08
    
    start_time = time.time()
    rng = np.random.default_rng(777)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, 2)
        new_point = points[idx] + noise
        
        # Clamp the new point
        new_point = _clamp_to_triangle_fast(new_point.reshape(1,2))[0]
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        new_score = _min_triangle_area_vec(points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            points[idx] = old_point
            
        # Cool down
        T *= alpha
        step_size *= 0.99998
        
    return best_points, best_score

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    h = np.sqrt(3) / 2
    configs = []
    
    # Configuration based on triangular lattice / known good patterns
    # Place points at vertices and then fill interior with quasi-regular pattern
    
    # 1. Near-optimal symmetric config (3 vertices + 3 edge midpoints + 5 interior)
    sym1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, 0.0],
        [0.25, h/2],
        [0.75, h/2],
        [0.5, h/3],
        [0.3, h/4],
        [0.7, h/4],
        [0.5, 2*h/3],
        [0.4, h/6]
    ])
    _clamp_to_triangle_fast(sym1)
    configs.append(sym1)
    
    # 2. Another symmetric arrangement
    sym2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.333, h/3],
        [0.667, h/3],
        [0.5, 2*h/3],
        [0.167, h/6],
        [0.833, h/6],
        [0.5, h/6],
        [0.25, h/2],
        [0.75, h/2]
    ])
    _clamp_to_triangle_fast(sym2)
    configs.append(sym2)
    
    # 3. Grid-based
    grid = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.25, h/3],
        [0.75, h/3],
        [0.5, 2*h/3],
        [0.125, h/6],
        [0.375, h/6],
        [0.625, h/6],
        [0.875, h/6],
        [0.5, h/3]
    ])
    _clamp_to_triangle_fast(grid)
    configs.append(grid)
    
    # 4. Vertex-centered with radial distribution
    cx, cy = 0.5, h/3  # centroid
    for r_frac in [0.3, 0.5, 0.7]:
        for angle_deg in [0, 60, 120, 180, 240, 300]:
            angle = np.radians(angle_deg)
            r = r_frac * h / 2
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            if len(configs) < 3 or True:
                pass
    
    # Better: generate structured configs
    # Config 4: 3 corners + 3 edge points + 5 interior in pentagon-like
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.5, 0.0],
        [0.25, h*0.5],
        [0.75, h*0.5],
        [0.5, h*0.333],
        [0.35, h*0.167],
        [0.65, h*0.167],
        [0.5, h*0.667],
        [0.4, h*0.4]
    ])
    _clamp_to_triangle_fast(cfg4)
    configs.append(cfg4)
    
    # Config 5: More spread out
    cfg5 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, h],
        [0.2, h*0.1],
        [0.8, h*0.1],
        [0.5, h*0.5],
        [0.1, h*0.3],
        [0.9, h*0.3],
        [0.35, h*0.7],
        [0.65, h*0.7],
        [0.5, h*0.2]
    ])
    _clamp_to_triangle_fast(cfg5)
    configs.append(cfg5)
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        pts[:, 1] *= h
        _clamp_to_triangle_fast(pts)
        configs.append(pts)
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 138  # 12s safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    configs = _generate_initial_configs()
    
    # Phase 1: Local optimization on each config
    n_configs = len(configs)
    
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        current_time_limit = min(3.0, time_left / max(1, n_configs - i))
        
        if current_time_limit < 0.5:
            break
            
        try:
            opt_pts, score = _local_optimize(config, time_limit=current_time_limit)
            if score > best_score:
                best_score = score
                best_points = opt_pts.copy()
        except Exception:
            continue
            
    # Phase 2: SA on the best found so far
    if best_points is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 2.0:
            try:
                opt_pts, score = _simulated_annealing(best_points, time_limit=remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
            except Exception:
                pass
                
    # Phase 3: Final local polish on the best
    if best_points is not None and time.time() < deadline - 1:
        remaining_time = deadline - time.time() - 1
        if remaining_time > 1.0:
            try:
                opt_pts, score = _local_optimize(best_points, time_limit=remaining_time)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = configs[0]
        
    return best_points

# EVOLVE-BLOCK-END
