# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    n = len(points)
    # Vectorized triangle area computation
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _negative_score(points):
    """Negative score for minimization."""
    s = _score_fast(points)
    if s <= 0:
        return 1e10
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    
    Uses a structural approach: start from symmetric configurations and use
    gradient-based optimization (BFGS) with multiple restarts from diverse
    initializations, including hexagonal, circular, and grid patterns.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    n = 13
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(seed=42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 13-gon (circle)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 2. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 3. Grid-based (4x4 minus one corner)
    grid_pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.5, 0.5], [1.5, 0.5],
        [0.5, 1.5], [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Two concentric circles
    inner_r = 0.4
    outer_r = 1.0
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer_angles = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 6, endpoint=False)
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    concentric = np.vstack([inner_pts, outer_pts])
    initializations.append(concentric)
    
    # 5. Random points in unit square (multiple seeds)
    for seed in [1, 2, 3, 4, 5]:
        rng_temp = np.random.default_rng(seed=seed)
        initializations.append(rng_temp.random((n, 2)))
    
    # 6. Perturbed regular polygon
    for perturb_scale in [0.1, 0.2, 0.3]:
        perturbed = reg_pts + rng.normal(size=(n, 2)) * perturb_scale
        initializations.append(perturbed)
    
    # Evaluate initializations and pick top ones
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        s = _score_fast(init_pts)
        scored_inits.append((s, init_pts))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Take top 5 initializations for optimization
    top_inits = [pts for _, pts in scored_inits[:5]]
    
    # Gradient-based optimization using BFGS
    for init_idx, init_pts in enumerate(top_inits):
        if time.time() > deadline - 30:
            break
        
        current = init_pts.copy()
        cur_score = _score_fast(current)
        update_best(current, cur_score)
        
        # Flatten to 1D for optimization
        def objective(flat_pts):
            pts = flat_pts.reshape(n, 2)
            return _negative_score(pts)
        
        x0 = current.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < _negative_score(current):
                new_pts = result.x.reshape(n, 2)
                new_score = _score_fast(new_pts)
                if new_score > cur_score:
                    current = new_pts
                    cur_score = new_score
                    update_best(current, cur_score)
        except Exception:
            pass
        
        # Second round of optimization from the result
        if time.time() > deadline - 30:
            break
        
        try:
            result2 = minimize(
                objective,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-8}
            )
            
            if result2.success or result2.fun < _negative_score(current):
                new_pts = result2.x.reshape(n, 2)
                new_score = _score_fast(new_pts)
                if new_score > cur_score:
                    current = new_pts
                    cur_score = new_score
                    update_best(current, cur_score)
        except Exception:
            pass
    
    # Additional random restarts from best point with small perturbations
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        perturbed = best_points + rng.normal(size=(n, 2)) * 0.05
        s = _score_fast(perturbed)
        if s > best_score:
            best_points = perturbed.copy()
            best_score = s
        
        # Quick optimization from here
        try:
            def objective(flat_pts):
                pts = flat_pts.reshape(n, 2)
                return _negative_score(pts)
            
            result = minimize(
                objective,
                best_points.flatten(),
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-7}
            )
            
            if result.success or result.fun < _negative_score(best_points):
                new_pts = result.x.reshape(n, 2)
                new_score = _score_fast(new_pts)
                if new_score > best_score:
                    best_points = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    # Final validation
    final_score = _score_fast(best_points)
    if final_score <= 0 or not np.all(np.isfinite(best_points)):
        # Fallback to a known valid configuration
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
