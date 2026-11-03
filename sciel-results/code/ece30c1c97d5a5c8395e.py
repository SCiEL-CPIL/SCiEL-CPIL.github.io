"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Uses trust-region reflective optimization on a smooth log-sum-exp surrogate of triangle areas,
with multi-start initialization (FPS, symmetric, random), followed by Nelder-Mead refinement
on the exact objective. Works in 2D barycentric coordinates (u, v) where w=1-u-v."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples (C(11,3) = 165)
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
N_TRIPLES = len(_IDX_TRIPLES)

def _bary_to_cart(bary2d):
    """Convert 2D barycentric (u, v) to Cartesian. w = 1-u-v. P = u*V0 + v*V1 + w*V2."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack([x, y])

def _cart_to_bary2d(pts):
    """Convert Cartesian to 2D barycentric (u, v)."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = 2.0 * y / np.sqrt(3)
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

def _clamp_bary(bary2d):
    """Project to valid barycentric simplex: u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary2d[:, 0])
    v = np.maximum(0.0, bary2d[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])

def _all_dets_from_bary(bary2d):
    """Compute all 165 determinants (proportional to area / TRI_AREA)."""
    u = bary2d[:, 0]
    v = bary2d[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det

def _min_area_ratio(bary2d):
    """Returns the minimum |det| (proportional to area / TRI_AREA)."""
    dets = _all_dets_from_bary(bary2d)
    return np.min(np.abs(dets))

def _hard_objective(x):
    """Hard objective: negative min |det|."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    return -np.min(np.abs(dets))

def _softmin_objective(x, beta=150.0):
    """Smooth surrogate: negative soft-min of |dets| using log-sum-exp."""
    bary2d = x.reshape(N, 2)
    bary2d = _clamp_bary(bary2d)
    dets = _all_dets_from_bary(bary2d)
    areas = np.abs(dets)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    # Numerically stable log-sum-exp
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a - log_sum / beta
    return -soft_min

def _fps_init_bary2d():
    """Farthest Point Sampling on a grid of barycentric coordinates."""
    n_grid = 25
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1.0/3, 1.0/3])]
    
    for _ in range(N - 1):
        sel_arr = np.array(selected)
        dists = np.min(np.linalg.norm(candidates[:, None, :] - sel_arr[None, :, :], axis=2), axis=1)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _symmetric_init_bary2d():
    """Generate a symmetric initial configuration in barycentric coords."""
    # Vertices, edge midpoints, centroid, and some interior points
    pts = np.array([
        [0.0, 0.0],   # V0
        [1.0, 0.0],   # V1
        [0.0, 1.0],   # V2
        [0.5, 0.0],   # midpoint V0-V1
        [0.5, 0.5],   # midpoint V1-V2
        [0.0, 0.5],   # midpoint V0-V2
        [1/3, 1/3],   # centroid
        [0.25, 0.15],
        [0.65, 0.15],
        [0.15, 0.55],
        [0.45, 0.35],
    ])
    return _clamp_bary(pts)

def _random_init_bary2d(rng):
    """Random initialization in barycentric coords."""
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0.02, 0.98)
            v = rng.uniform(0.02, 0.98)
            if u + v < 0.96:
                bary[i] = [u, v]
                break
    return bary

def _structured_init_bary2d(seed=42):
    """Generate structured initial configurations in barycentric coords."""
    rng = np.random.RandomState(seed)
    
    patterns = []
    
    # Pattern A: Symmetric with vertices, edge midpoints, centroid
    pA = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1/3, 1/3],
        [0.2, 0.1], [0.8, 0.1], [0.1, 0.7], [0.5, 0.3]
    ])
    patterns.append(pA)
    
    # Pattern B: Concentric hexagons
    pB = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    patterns.append(pB)
    
    # Pattern C: More spread
    pC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.4, 0.0], [0.6, 0.4], [0.0, 0.4],
        [0.33, 0.33],
        [0.1, 0.05], [0.9, 0.05], [0.05, 0.9], [0.5, 0.15]
    ])
    patterns.append(pC)
    
    # Pattern D: Hexagonal lattice
    pD = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    patterns.append(pD)
    
    idx = seed % len(patterns)
    bary = patterns[idx].copy()
    jitter = rng.normal(0, 0.015, size=bary.shape)
    bary = bary + jitter
    bary = _clamp_bary(bary)
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    best_bary2d = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initial_bary_list = []
    
    # FPS initialization
    if time_left() > 30.0:
        bary_fps = _fps_init_bary2d()
        initial_bary_list.append(bary_fps)
    
    # Symmetric initialization
    if time_left() > 30.0:
        bary_sym = _symmetric_init_bary2d()
        initial_bary_list.append(bary_sym)
    
    # Structured initializations
    for seed in [42, 123, 456, 789]:
        if time_left() < 25.0:
            break
        bary_struct = _structured_init_bary2d(seed)
        initial_bary_list.append(bary_struct)
    
    # Random initializations
    n_random = 6
    for i in range(n_random):
        if time_left() < 20.0:
            break
        bary_rand = _random_init_bary2d(rng)
        initial_bary_list.append(bary_rand)
    
    # Optimize each initialization
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for start_bary in initial_bary_list:
        if time_left() < 15.0:
            break
        
        try:
            # Phase 1: Trust-region reflective with soft-min surrogate (differentiable)
            x0 = start_bary.flatten()
            
            def obj_soft(x, beta=150.0):
                return _softmin_objective(x, beta=beta)
            
            res1 = minimize(
                obj_soft, x0, method='trust-constr',
                bounds=bounds,
                options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
            )
            bary_opt1 = _clamp_bary(res1.x.reshape(N, 2))
            score1 = _min_area_ratio(bary_opt1)
            
            if score1 > best_score:
                best_score = score1
                best_bary2d = bary_opt1.copy()
            
            # Phase 2: Refine with hard objective using L-BFGS-B
            if time_left() > 12.0:
                res2 = minimize(
                    _hard_objective, bary_opt1.flatten(), method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                bary_opt2 = _clamp_bary(res2.x.reshape(N, 2))
                score2 = _min_area_ratio(bary_opt2)
                
                if score2 > best_score:
                    best_score = score2
                    best_bary2d = bary_opt2.copy()
                    
        except Exception:
            pass
    
    # Polish best solution with Nelder-Mead on hard objective
    if best_bary2d is not None and time_left() > 20.0:
        try:
            res_nm = minimize(
                _hard_objective, best_bary2d.flatten(), method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            bary_nm = _clamp_bary(res_nm.x.reshape(N, 2))
            score_nm = _min_area_ratio(bary_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_bary2d = bary_nm.copy()
        except Exception:
            pass
    
    # Additional refinement from perturbed best solution
    if best_bary2d is not None and time_left() > 30.0:
        for i in range(5):
            if time_left() < 15.0:
                break
            try:
                noise = rng.normal(0, 0.005, size=2 * N)
                perturbed = _clamp_bary(best_bary2d + noise)
                
                res_p = minimize(
                    _hard_objective, perturbed.flatten(), method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                bary_p = _clamp_bary(res_p.x.reshape(N, 2))
                score_p = _min_area_ratio(bary_p)
                
                if score_p > best_score:
                    best_score = score_p
                    best_bary2d = bary_p.copy()
            except Exception:
                pass
    
    # Convert best barycentric to Cartesian
    if best_bary2d is None:
        # Fallback: use FPS
        bary2d = _fps_init_bary2d()
    else:
        bary2d = _clamp_bary(best_bary2d)
    
    pts = _bary_to_cart(bary2d)
    
    return np.array(pts)
