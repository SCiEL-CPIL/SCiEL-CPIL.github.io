"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

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

def _softmin_objective(x, beta=80.0):
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

def _softmin_gradient(x, beta=80.0):
    """Gradient of the soft-min objective."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    idx = _get_indices()
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    shifted = areas - min_area
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft_min w.r.t. areas: -weights
    # Gradient of areas w.r.t. points
    # area = 0.5 * |cross|, cross = (pj-pi) x (pk-pi)
    # d(area)/d(pi) = -0.5 * sign(cross) * ((pk-pi) perp - (pj-pi) perp) ... complex
    # For simplicity, use numerical gradient or just return the soft-min value
    # Let scipy handle it
    return None

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

def _nm_objective(x):
    """Objective for Nelder-Mead: negative min area, with projection."""
    bary = x.reshape(-1, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    return -_min_triangle_area_vectorized(points)

def _coordinate_descent_refine(bary, max_iter=50, step=0.01):
    """Fast coordinate descent refinement in barycentric coordinates."""
    bary = bary.copy()
    best_score = _min_triangle_area_vectorized(_bary_to_cart(bary))
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations
                for delta in [step, -step, step*0.5, -step*0.5]:
                    bary_trial = bary.copy()
                    bary_trial[i, coord] += delta
                    bary_trial = _clamp_bary(bary_trial)
                    score_trial = _min_triangle_area_vectorized(_bary_to_cart(bary_trial))
                    if score_trial > best_score + 1e-12:
                        bary = bary_trial
                        best_score = score_trial
                        improved = True
                        break
        if not improved:
            break
    
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
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
    
    # Generate multiple initializations: grid, random, perturbed
    inits = [bary_init]
    
    # Add random initializations
    for seed in range(1, 15):
        if time_left() < 3.0:
            break
        bary_r = _make_initial_bary_random(seed)
        inits.append(bary_r)
        
        # Also add perturbed versions of the grid
        if seed <= 8:
            rng_p = np.random.RandomState(seed + 100)
            for scale in [0.03, 0.08, 0.15]:
                if time_left() < 3.0:
                    break
                bary_p = _make_initial_bary_perturb(bary_init, rng_p, scale)
                inits.append(bary_p)
    
    # Optimize each initialization
    for i, bary0 in enumerate(inits):
        if time_left() < 8.0:
            break
            
        x0 = bary0.flatten()
        
        # L-BFGS-B with soft-min surrogate (higher beta for tighter approximation)
        if time_left() > 15.0:
            try:
                result_lbfgs = minimize(
                    _softmin_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                bary_opt = _clamp_bary(result_lbfgs.x.reshape(-1, 2))
                points_opt = _bary_to_cart(bary_opt)
                score_opt = _min_triangle_area_vectorized(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
                    best_bary = bary_opt.copy()
            except Exception:
                pass
        
        # Coordinate descent refinement (fast, no gradient needed)
        if time_left() > 5.0 and best_bary is not None:
            try:
                bary_cd = _coordinate_descent_refine(best_bary.copy(), max_iter=30, step=0.005)
                points_cd = _bary_to_cart(bary_cd)
                score_cd = _min_triangle_area_vectorized(points_cd)
                if score_cd > best_score:
                    best_score = score_cd
                    best_points = points_cd.copy()
                    best_bary = bary_cd.copy()
            except Exception:
                pass
    
    # Final polish: multiple L-BFGS passes with increasing beta
    if time_left() > 10.0 and best_bary is not None:
        for beta in [100.0, 150.0]:
            if time_left() < 8.0:
                break
            try:
                x_final = best_bary.flatten()
                result_final = minimize(
                    lambda x, b=beta: _softmin_objective(x, beta=b),
                    x_final,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                bary_final = _clamp_bary(result_final.x.reshape(-1, 2))
                points_final = _bary_to_cart(bary_final)
                score_final = _min_triangle_area_vectorized(points_final)
                if score_final > best_score:
                    best_score = score_final
                    best_points = points_final.copy()
                    best_bary = bary_final.copy()
            except Exception:
                pass
        
        # One more coordinate descent pass
        if time_left() > 5.0:
            try:
                bary_cd = _coordinate_descent_refine(best_bary.copy(), max_iter=50, step=0.003)
                points_cd = _bary_to_cart(bary_cd)
                score_cd = _min_triangle_area_vectorized(points_cd)
                if score_cd > best_score:
                    best_score = score_cd
                    best_points = points_cd.copy()
                    best_bary = bary_cd.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _bary_to_cart(bary_init)
    
    return best_points
