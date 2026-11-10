# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

# Precompute triangle indices for 13 points
_N = 13
_I_IDX, _J_IDX, _K_IDX = [], [], []
for _i in range(_N):
    for _j in range(_i + 1, _N):
        for _k in range(_j + 1, _N):
            _I_IDX.append(_i)
            _J_IDX.append(_j)
            _K_IDX.append(_k)
_I_IDX = np.array(_I_IDX)
_J_IDX = np.array(_J_IDX)
_K_IDX = np.array(_K_IDX)

def _all_triangle_areas(pts):
    """Compute all 286 triangle areas vectorized."""
    x1, y1 = pts[_I_IDX, 0], pts[_I_IDX, 1]
    x2, y2 = pts[_J_IDX, 0], pts[_J_IDX, 1]
    x3, y3 = pts[_K_IDX, 0], pts[_K_IDX, 1]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _exact_score(pts):
    """Compute A_min / A_hull exactly."""
    if len(pts) < 3:
        return 0.0
    areas = _all_triangle_areas(pts)
    a_min = np.min(areas)
    a_hull = _hull_area(pts)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull

def _normalize_hull_area(pts, target_area=1.0):
    """Scale points so that the convex hull area is target_area."""
    a = _hull_area(pts)
    if a <= 0 or not np.isfinite(a):
        return pts
    scale = np.sqrt(target_area / a)
    return pts * scale

def _relaxation_step(pts, laplacian_weight=1.0, repulsion_weight=1.0, step_size=0.05, k_neighbors=4):
    """
    Perform one step of physics-based relaxation.
    - Laplacian smoothing: Pulls points towards the average of their k-nearest neighbors.
    - Repulsion: Pushes points away from each other.
    """
    n = len(pts)
    # Compute distances to find k-nearest neighbors
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Find k-nearest neighbors for each point
    # Exclude self
    np.fill_diagonal(dists, np.inf)
    neighbor_indices = np.argsort(dists, axis=1)[:, :k_neighbors]
    
    forces = np.zeros_like(pts)
    
    # Laplacian force: sum of (neighbor - current)
    for i in range(n):
        neighbors = pts[neighbor_indices[i]]
        forces[i] += laplacian_weight * (neighbors.mean(axis=0) - pts[i])
        
    # Repulsion force: sum of (current - other) / dist^2
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d_vec = pts[i] - pts[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-6:
                # Random small push if coincident
                forces[i] += repulsion_weight * np.random.normal(0, 0.01, 2)
            else:
                # Force is proportional to 1/d^2, direction away from j
                forces[i] += repulsion_weight * (d_vec / (d_norm * d_norm))
                
    # Update positions
    new_pts = pts + step_size * forces
    
    # Normalize to keep hull area constant (prevents scaling drift)
    new_pts = _normalize_hull_area(new_pts, 1.0)
    
    return new_pts

def _polish(pts, time_limit=10.0):
    """Polish the configuration using Nelder-Mead on the exact score."""
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        p = x.reshape(13, 2)
        if not np.all(np.isfinite(p)):
            return 1e10
        s = _exact_score(p)
        if s <= 0:
            return 1e10
        return -s
    
    try:
        result = minimize(
            obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        if np.all(np.isfinite(result.x)):
            return result.x.reshape(13, 2)
    except Exception:
        pass
    return pts

def _generate_initial_configs():
    """Generate a diverse set of initial configurations."""
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric rings: 1 center, 5 inner, 7 outer? No, 1+5+7=13.
    # Let's try 1 center, 6 inner (hexagon), 6 outer (hexagon)
    for r_inner in [0.3, 0.4, 0.5]:
        for r_outer in [0.8, 1.0, 1.2]:
            for a_off in [0, np.pi/6]:
                center = np.array([0.0, 0.0])
                inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + a_off
                inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
                outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + a_off + np.pi/6 # offset
                outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
                pts = np.vstack([center.reshape(1,2), inner_pts, outer_pts])
                configs.append(_normalize_hull_area(pts, 1.0))
                
    # 3. Triangular lattice fragment
    # 3 rows: 1, 2, 3, 4, 3? No, 13 points.
    # Rows of 2, 3, 4, 3, 1?
    # Let's just use random perturbations of the regular polygon
    for _ in range(20):
        base = configs[0]
        perturbed = base + rng.normal(0, 0.1, base.shape)
        configs.append(_normalize_hull_area(perturbed, 1.0))
        
    # 4. Random points in unit disk, projected to hull?
    for _ in range(10):
        pts = rng.uniform(-1, 1, (13, 2))
        # Ensure convex hull is valid
        try:
            ConvexHull(pts)
            configs.append(_normalize_hull_area(pts, 1.0))
        except Exception:
            pass
            
    return configs

def heilbronn_convex13() -> np.ndarray:
    """
    Find 13 points maximizing A_min / A_hull.
    Uses physics-based relaxation (Laplacian + Repulsion) followed by Nelder-Mead polishing.
    """
    start_time = time.time()
    time_limit = 580  # Safety margin from 600s limit
    
    def time_check(remaining=20):
        return time.time() - start_time < time_limit - remaining
        
    best_pts = None
    best_score = 0.0
    
    def update_best(pts):
        nonlocal best_pts, best_score
        if pts is None or not np.all(np.isfinite(pts)) or len(pts) != 13:
            return
        score = _exact_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Phase 1: Generate and evaluate initial configs
    initial_configs = _generate_initial_configs()
    
    for pts in initial_configs:
        if not time_check(100):
            break
        update_best(pts)
        
    # Phase 2: Relaxation and Polishing for top candidates
    # Rank initial configs by score
    scored_configs = []
    for pts in initial_configs:
        score = _exact_score(pts)
        scored_configs.append((score, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Take top 10 configurations
    top_configs = [cfg for _, cfg in scored_configs[:10]]
    
    for init_pts in top_configs:
        if not time_check(60):
            break
            
        pts = init_pts.copy()
        
        # Relaxation Loop
        # Adaptive step size: start large, decrease
        steps = 200
        for i in range(steps):
            if not time_check(30):
                break
                
            # Annealing schedule for step size
            step_size = 0.05 * (1.0 - i / steps) + 0.001
            # Increase repulsion weight slightly to ensure separation
            rep_weight = 0.5 + 0.5 * (i / steps)
            
            pts = _relaxation_step(pts, laplacian_weight=1.0, repulsion_weight=rep_weight, step_size=step_size, k_neighbors=4)
            
            # Periodically check score and update best
            if i % 20 == 0:
                update_best(pts)
                
        # Polish the relaxed configuration
        if time_check(20):
            polished_pts = _polish(pts, time_limit=15.0)
            update_best(polished_pts)
            
    # Phase 3: Multi-start from best found so far
    if time_check(30) and best_pts is not None:
        rng = np.random.default_rng(123)
        for trial in range(5):
            if not time_check(20):
                break
                
            # Perturb best solution
            noise_scale = 0.05
            perturbed = best_pts + rng.normal(0, noise_scale, best_pts.shape)
            perturbed = _normalize_hull_area(perturbed, 1.0)
            
            # Quick relaxation
            pts = perturbed
            for i in range(50):
                if not time_check(10):
                    break
                step_size = 0.01
                pts = _relaxation_step(pts, laplacian_weight=1.0, repulsion_weight=1.0, step_size=step_size, k_neighbors=4)
                
            # Polish
            if time_check(10):
                polished_pts = _polish(pts, time_limit=10.0)
                update_best(polished_pts)
                
    # Final fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)) or len(best_pts) != 13:
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts

# EVOLVE-BLOCK-END
