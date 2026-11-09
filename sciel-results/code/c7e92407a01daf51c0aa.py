import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def get_full_points(reduced_params):
    """
    Reconstruct 11x2 points from 11 reduced parameters.
    Structure:
    - Point 0: (0.5, cy)
    - Points 1-5: (x_i, y_i) in right half (x >= 0.5)
    - Points 6-10: (1-x_i, y_i) in left half (x <= 0.5)
    """
    cy = reduced_params[0]
    pts = np.empty((11, 2))
    pts[0] = [0.5, cy]
    
    right_pts = reduced_params[1:].reshape(5, 2)
    pts[1:6] = right_pts
    pts[6:11] = np.column_stack([1.0 - right_pts[:, 0], right_pts[:, 1]])
    
    return pts

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def is_inside_reduced(reduced_params):
    """
    Check if the reduced points correspond to a valid configuration inside the triangle.
    """
    cy = reduced_params[0]
    if cy < -1e-9 or cy > H + 1e-9:
        return False
        
    right_pts = reduced_params[1:].reshape(5, 2)
    x = right_pts[:, 0]
    y = right_pts[:, 1]
    
    if np.any(x < 0.5 - 1e-9):
        return False
    if np.any(x > 1.0 + 1e-9):
        return False
    if np.any(y < -1e-9):
        return False
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective_exact(params):
    """
    Objective for local refinement using exact min.
    """
    if not is_inside_reduced(params):
        return 1.0
        
    points = get_full_points(params)
    area = min_triangle_area_vec(points)
    return -area

def objective_da(params):
    """
    Objective for dual-annealing with penalty for out-of-bounds.
    """
    if not is_inside_reduced(params):
        return 1.0
    points = get_full_points(params)
    area = min_triangle_area_vec(points)
    return -area

def init_symmetric():
    """
    Symmetric initialization: points arranged in a symmetric pattern.
    """
    params = np.zeros(11)
    # Center point
    params[0] = H / 3.0
    # 5 right-side points in a grid-like pattern
    xs = np.array([0.55, 0.7, 0.85, 0.6, 0.75])
    ys = np.array([H * 0.1, H * 0.3, H * 0.15, H * 0.5, H * 0.25])
    for i in range(5):
        params[1 + 2 * i] = xs[i]
        params[2 + 2 * i] = ys[i]
    return params

def init_lattice():
    """
    Lattice-based initialization.
    """
    params = np.zeros(11)
    params[0] = H / 2.5
    # Place points in a hexagonal-like lattice
    xs = np.array([0.55, 0.65, 0.75, 0.85, 0.6])
    ys = np.array([H * 0.15, H * 0.35, H * 0.1, H * 0.45, H * 0.55])
    for i in range(5):
        params[1 + 2 * i] = xs[i]
        params[2 + 2 * i] = ys[i]
    return params

def init_random(rng):
    """
    Random initialization inside the triangle.
    """
    cy = rng.uniform(0.1 * H, 0.9 * H)
    params = np.zeros(11)
    params[0] = cy
    for i in range(5):
        x = rng.uniform(0.5, 1.0)
        max_y = np.sqrt(3) * (1 - x)
        y = rng.uniform(0.05 * max_y, 0.95 * max_y)
        params[1 + 2 * i] = x
        params[2 + 2 * i] = y
    return params

def nelder_mead_refine(params, deadline, max_iter=2000):
    """
    Nelder-Mead refinement from a single starting point.
    """
    if time.time() > deadline:
        return params
    
    best_params = params.copy()
    best_score = -objective_exact(best_params)
    
    try:
        res = minimize(
            objective_exact,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        if is_inside_reduced(res.x):
            new_score = -res.fun
            if new_score > best_score:
                best_score = new_score
                best_params = res.x.copy()
    except Exception:
        pass
    
    return best_params

def refine_all_points(params, deadline):
    """
    Multi-start local refinement.
    """
    if time.time() > deadline:
        return params
    
    best_params = params.copy()
    best_score = -objective_exact(best_params)
    
    rng = np.random.default_rng(123)
    
    # First: direct Nelder-Mead
    candidate = nelder_mead_refine(best_params, deadline, max_iter=3000)
    if is_inside_reduced(candidate):
        score = -objective_exact(candidate)
        if score > best_score:
            best_score = score
            best_params = candidate.copy()
    
    # Then: perturbed starts
    for attempt in range(30):
        if time.time() > deadline - 2:
            break
        
        # Random perturbation
        sigma = 0.01 * (1.0 - 0.5 * attempt / 30)
        perturbation = rng.normal(0, sigma, size=best_params.shape)
        x0 = best_params + perturbation
        
        # Clip to bounds
        x0[0] = np.clip(x0[0], 0.0, H)
        for i in range(5):
            x0[1 + 2 * i] = np.clip(x0[1 + 2 * i], 0.5, 1.0)
            x0[2 + 2 * i] = np.clip(x0[2 + 2 * i], 0.0, H)
        
        if not is_inside_reduced(x0):
            continue
        
        candidate = nelder_mead_refine(x0, deadline, max_iter=1500)
        if is_inside_reduced(candidate):
            score = -objective_exact(candidate)
            if score > best_score:
                best_score = score
                best_params = candidate.copy()
    
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    bounds = [(0.0, H)]
    for i in range(5):
        bounds.append((0.5, 1.0))
        bounds.append((0.0, H))
    
    best_params = None
    best_score = -float('inf')
    
    def update_best(params):
        nonlocal best_params, best_score
        if is_inside_reduced(params):
            score = -objective_exact(params)
            if score > best_score:
                best_score = score
                best_params = params.copy()
    
    # Phase 1: Multiple diverse initializations + local refinement
    inits = [
        init_symmetric(),
        init_lattice(),
    ]
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(init_random(rng))
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        update_best(init)
    
    # Phase 2: Dual-annealing global search
    if time.time() < deadline - 30:
        try:
            da_result = dual_annealing(
                objective_da,
                bounds=bounds,
                maxiter=300,
                seed=42,
                restarts=10,
                polish=True,
                initial_temp=5230.0,
                visit='2-furthest',
                acceptance='lamarkian'
            )
            if is_inside_reduced(da_result.x):
                update_best(da_result.x)
        except Exception:
            pass
    
    # Phase 3: Differential evolution with exact objective
    if time.time() < deadline - 20:
        try:
            de_result = differential_evolution(
                objective_exact,
                bounds,
                maxiter=200,
                popsize=20,
                tol=0.001,
                seed=42,
                polish=False,
                init='sobol'
            )
            if is_inside_reduced(de_result.x):
                update_best(de_result.x)
        except Exception:
            pass
    
    # Phase 4: Aggressive local refinement
    if best_params is not None and time.time() < deadline - 5:
        best_params = refine_all_points(best_params, deadline)
    
    if best_params is None:
        # Fallback
        best_params = init_symmetric()
    
    return get_full_points(best_params)
