"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0  # area of the unit-base equilateral triangle

# Precompute indices for all triples to speed up area calculation
_IDX_TRIPLES = None
def _get_indices():
    global _IDX_TRIPLES
    if _IDX_TRIPLES is None:
        n = N
        _IDX_TRIPLES = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    return _IDX_TRIPLES

def _min_triangle_area_vectorized(points):
    """Compute the smallest area among all triangles formed by triples of points, vectorized."""
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    x = bary[:, 0] * V1[0] + bary[:, 1] * V2[0]
    y = bary[:, 0] * V1[1] + bary[:, 1] * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V1 + v*V2."""
    u = points[:, 0] - 0.5 * points[:, 1]
    v = 2.0 * points[:, 1] / np.sqrt(3)
    return np.column_stack([u, v])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    bary = np.clip(bary, 0.0, None)
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1.0
    if np.any(mask):
        bary[mask] = bary[mask] / s[mask, None]
    return bary

def _softmin_objective(x, beta=50.0):
    """Smooth surrogate for the min triangle area using soft-min (log-sum-exp based)."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    min_area = np.min(areas)
    shifted = areas - min_area
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = min_area - log_sum
    
    return -soft_min

def _objective_bary(x):
    """Objective: negative min triangle area, with points in barycentric coords."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _make_initial_bary_grid(seed):
    """Generate a structured grid-based initial barycentric configuration."""
    rng = np.random.RandomState(seed)
    
    # Key structural points: vertices, edge midpoints, centroid
    bary_list = [
        [0.0, 0.0],       # V0
        [1.0, 0.0],       # V1
        [0.0, 1.0],       # V2
        [0.5, 0.0],       # midpoint V0-V1
        [0.5, 0.5],       # midpoint V1-V2
        [0.0, 0.5],       # midpoint V2-V0
        [1.0/3, 1.0/3],   # centroid
    ]
    
    # Add interior points in a regular pattern
    additional = [
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.75],
        [0.0, 0.25],
        [0.0, 0.75],
    ]
    
    for pt in additional:
        if len(bary_list) < N:
            if pt[0] >= 0 and pt[1] >= 0 and pt[0] + pt[1] <= 1:
                bary_list.append(pt)
    
    while len(bary_list) < N:
        u = rng.uniform(0.1, 0.8)
        v = rng.uniform(0.1, 0.8)
        if u + v < 0.85:
            bary_list.append([u, v])
    
    return np.array(bary_list[:N])

def _make_initial_bary_random(seed):
    """Generate random initial barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(500):
        u = rng.uniform(0.02, 0.98)
        v = rng.uniform(0.02, 0.98)
        if u + v < 0.96:
            bary_list.append([u, v])
            if len(bary_list) == N:
                break
    while len(bary_list) < N:
        u = rng.uniform(0.05, 0.8)
        v = rng.uniform(0.05, 0.8)
        if u + v < 0.9:
            bary_list.append([u, v])
    return np.array(bary_list[:N])

def _make_initial_bary_perturb(base_bary, rng, scale):
    """Perturb a base barycentric configuration."""
    bary_pert = base_bary + rng.normal(0, scale, size=base_bary.shape)
    bary_pert = _clamp_bary(bary_pert)
    return bary_pert

def _local_optimize(x0, bounds, maxiter=200, gtol=1e-9, use_soft=False, beta=50.0):
    """Run local optimization from x0."""
    try:
        if use_soft:
            res = minimize(
                lambda x: _softmin_objective(x, beta), x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'gtol': gtol}
            )
        else:
            res = minimize(
                _objective_bary, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'gtol': gtol}
            )
        return res.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    best_bary = None
    
    def time_left():
        return deadline - time.time()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Start with a structured grid-based initialization
    bary_init = _make_initial_bary_grid(0)
    points_init = _bary_to_cart(bary_init)
    score_init = _min_triangle_area_vectorized(points_init)
    best_score = score_init
    best_points = points_init.copy()
    best_bary = bary_init.copy()
    
    # Phase 1: Soft-min optimization from the grid start (smooth surrogate)
    x0 = bary_init.flatten()
    x_soft = _local_optimize(x0, bounds, maxiter=300, gtol=1e-10, use_soft=True, beta=30.0)
    bary_soft = _clamp_bary(x_soft.reshape(-1, 2))
    points_soft = _bary_to_cart(bary_soft)
    score_soft = _min_triangle_area_vectorized(points_soft)
    
    if score_soft > best_score:
        best_score = score_soft
        best_points = points_soft.copy()
        best_bary = bary_soft.copy()
    
    # Phase 2: Standard local optimization from the grid start
    x_opt = _local_optimize(x0, bounds, maxiter=300, gtol=1e-10)
    bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
    points_opt = _bary_to_cart(bary_opt)
    score_opt = _min_triangle_area_vectorized(points_opt)
    
    if score_opt > best_score:
        best_score = score_opt
        best_points = points_opt.copy()
        best_bary = bary_opt.copy()
    
    # Phase 3: Multi-restart with perturbations and random starts
    rng = np.random.RandomState(42)
    
    max_restarts = 500
    restart = 0
    while restart < max_restarts and time_left() > 2.0:
        restart += 1
        
        if restart % 7 == 0:
            # Random restart
            bary_start = _make_initial_bary_random(restart * 1000)
        elif restart % 2 == 0:
            # Perturb best solution with moderate scale
            scale = 0.05 + 0.05 * (1.0 - restart / max_restarts)
            bary_start = _make_initial_bary_perturb(best_bary, rng, scale)
        else:
            # Perturb best solution with larger scale
            scale = 0.1 + 0.1 * (1.0 - restart / max_restarts)
            bary_start = _make_initial_bary_perturb(best_bary, rng, scale)
        
        # Alternate between soft-min and standard optimization
        use_soft = (restart % 3 == 0)
        beta = 20.0 + 20.0 * (restart % 5) / 5.0
        
        x0 = bary_start.flatten()
        x_opt = _local_optimize(x0, bounds, maxiter=150, gtol=1e-9, use_soft=use_soft, beta=beta)
        bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
        points_opt = _bary_to_cart(bary_opt)
        score_opt = _min_triangle_area_vectorized(points_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt.copy()
            best_bary = bary_opt.copy()
    
    # Phase 4: Fine-tuning from the best solution found
    if time_left() > 3.0 and best_bary is not None:
        # Try multiple fine-tuning runs from the best with very small perturbations
        for ft in range(8):
            if time_left() < 2.0:
                break
            scale = 0.01 + 0.01 * (ft / 8.0)
            bary_ft = _make_initial_bary_perturb(best_bary, rng, scale)
            x0 = bary_ft.flatten()
            
            # Use standard objective for high-precision polish
            x_opt = _local_optimize(x0, bounds, maxiter=500, gtol=1e-11)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
        
        # Final polish with even tighter tolerance
        if time_left() > 2.0:
            x0 = best_bary.flatten()
            x_opt = _local_optimize(x0, bounds, maxiter=1000, gtol=1e-12)
            bary_opt = _clamp_bary(x_opt.reshape(-1, 2))
            points_opt = _bary_to_cart(bary_opt)
            score_opt = _min_triangle_area_vectorized(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                best_bary = bary_opt.copy()
    
    # Final fallback
    if best_points is None:
        bary0 = _make_initial_bary_grid(0)
        best_points = _bary_to_cart(bary0)
    
    return np.array(best_points)
