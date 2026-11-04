"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Precompute triple indices for efficiency
_triple_indices = np.array(list(combinations(range(N), 3)))
_TRI_I = _triple_indices[:, 0]
_TRI_J = _triple_indices[:, 1]
_TRI_K = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min(ratios, tau):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -1/tau * log(sum(exp(-tau * x)))
    """
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -100, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _transform_to_fixed(pts):
    """
    Transform point set so that p[0]=(0,0), p[1]=(1,0).
    Returns the transformed points (13, 2).
    """
    pts = pts.copy()
    pts -= pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return pts
    pts /= length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts

def _params_to_pts(params):
    """Convert 22-dim param vector to (13,2) point array."""
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _exact_objective(params, tau):
    """
    Objective: minimize -smooth_min(triangle_area / hull_area).
    Uses a tighter formulation that normalizes by hull area internally.
    """
    pts = _params_to_pts(params)
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 1e9
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _grid_based_init(rng, idx):
    """
    Grid-based initialization: sample from a structured grid of regular
    polygon perturbations. This is structurally superior for small N.
    """
    # Use a set of structured starting configurations
    # For N=13, the optimal is close to a regular 13-gon
    
    # Strategy: take regular 13-gon, apply structured perturbations
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Different perturbation patterns based on idx
    patterns = [
        'uniform_small', 'alternating', 'radial', 'angle_jitter',
        'pair_swap', 'sector', 'spiral', 'lattice',
        'arc_cluster', 'zigzag', 'bump', 'twist'
    ]
    pattern = patterns[idx % len(patterns)]
    
    if pattern == 'uniform_small':
        angles = base_angles + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.05, N)
    elif pattern == 'alternating':
        angles = base_angles + np.where(np.arange(N) % 2 == 0, 0.08, -0.08)
        radii = 1.0 + np.where(np.arange(N) % 2 == 0, 0.06, -0.06)
    elif pattern == 'radial':
        angles = base_angles
        radii = 1.0 + rng.normal(0, 0.12, N)
    elif pattern == 'angle_jitter':
        angles = base_angles + rng.uniform(-0.2, 0.2, N)
        radii = 1.0
    elif pattern == 'pair_swap':
        angles = base_angles.copy()
        for _ in range(3):
            i, j = rng.integers(0, N, 2)
            if i != j:
                angles[i], angles[j] = angles[j], angles[i]
        angles = np.sort(angles)
        radii = 1.0 + rng.normal(0, 0.05, N)
    elif pattern == 'sector':
        angles = base_angles + rng.uniform(0, 0.15, N)
        radii = 1.0 + rng.uniform(0, 0.1, N)
    elif pattern == 'spiral':
        t = np.arange(N) / N
        angles = base_angles + 0.15 * t
        radii = 1.0 + 0.1 * t
    elif pattern == 'lattice':
        # Points near a 2D lattice pattern mapped to circle
        angles = base_angles + rng.normal(0, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.08, N)
    elif pattern == 'arc_cluster':
        angles = base_angles + rng.normal(0, 0.15, N)
        radii = 1.0 + np.concatenate([rng.normal(0, 0.1, 6), rng.normal(0, 0.1, 7)])
    elif pattern == 'zigzag':
        angles = base_angles + 0.07 * np.sin(np.arange(N) * np.pi / 3)
        radii = 1.0 + 0.07 * np.cos(np.arange(N) * np.pi / 4)
    elif pattern == 'bump':
        angles = base_angles
        radii = 1.0 + 0.12 * np.exp(-0.5 * ((np.arange(N) - 3) / 2.0) ** 2)
    elif pattern == 'twist':
        angles = base_angles + 0.1 * np.sin(2 * np.pi * np.arange(N) / N)
        radii = 1.0 + 0.08 * np.cos(2 * np.pi * np.arange(N) / N)
    else:
        angles = base_angles + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.normal(0, 0.05, N)
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Transform to fixed frame
    pts = _transform_to_fixed(pts)
    return pts[2:].flatten()

def _local_search_step(params, rng, tau=500, n_steps=50):
    """
    Coordinate-wise local search: perturb individual coordinates and accept
    if score improves. More targeted than DE for this problem structure.
    """
    current_pts = _params_to_pts(params)
    current_score = _compute_min_ratio(current_pts)
    
    best_params = params.copy()
    best_score = current_score
    
    step_size = 0.02
    for step in range(n_steps):
        # Pick a random coordinate to perturb
        coord_idx = rng.integers(0, 2 * (N - 2))
        point_idx = coord_idx // 2 + 2  # points 2..12
        axis = coord_idx % 2
        
        # Try a few step sizes
        for ss in [step_size, step_size * 0.5, step_size * 0.25]:
            for direction in [1, -1]:
                trial = best_params.copy()
                trial[coord_idx] += direction * ss
                
                trial_pts = _params_to_pts(trial)
                trial_score = _compute_min_ratio(trial_pts)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_params = trial.copy()
        
        # Adapt step size
        if best_score > current_score + 1e-6:
            step_size *= 1.1
        else:
            step_size *= 0.9
        step_size = np.clip(step_size, 0.001, 0.1)
    
    return best_params, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse structured starting points
    num_starts = 16
    starts = []
    for i in range(num_starts):
        if (time.time() - start_time) > time_limit - 25.0:
            break
        p = _grid_based_init(rng, i)
        starts.append(p)
    
    # Also add a couple of pure polygon starts
    for i in range(2):
        if (time.time() - start_time) > time_limit - 25.0:
            break
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.05, 0.05, N)
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        pts = _transform_to_fixed(pts)
        starts.append(pts[2:].flatten())
    
    # Phase 2: Smooth-min optimization with annealing tau
    tau_schedule = [20, 60, 150, 400, 1000]
    
    for tau in tau_schedule:
        if (time.time() - start_time) > time_limit - 10.0:
            break
        
        for i in range(len(starts)):
            if (time.time() - start_time) > time_limit - 6.0:
                break
            
            try:
                res = minimize(
                    _exact_objective,
                    starts[i],
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                new_params = res.x
                pts = _params_to_pts(new_params)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = new_params.copy()
                
                starts[i] = new_params.copy()
                
            except Exception:
                pass
    
    # Phase 3: High-tau polish on top solutions
    if best_params is not None and (time.time() - start_time) < time_limit - 20.0:
        # Sort starts by score, take top 4
        scored_starts = []
        for s in starts:
            pts = _params_to_pts(s)
            sc = _compute_min_ratio(pts)
            scored_starts.append((sc, s))
        scored_starts.sort(key=lambda x: -x[0])
        
        for sc, s in scored_starts[:4]:
            if (time.time() - start_time) > time_limit - 8.0:
                break
            try:
                res = minimize(
                    _exact_objective,
                    s,
                    args=(3000,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                pts = _params_to_pts(res.x)
                score = _compute_min_ratio(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 4: Coordinate-wise local search (structural change from DE)
    if best_params is not None and (time.time() - start_time) < time_limit - 15.0:
        # Run local search on best and a few nearby
        candidates = [best_params]
        for s in scored_starts[:3]:
            candidates.append(s[1].copy())
        
        for cand in candidates:
            if (time.time() - start_time) > time_limit - 8.0:
                break
            try:
                improved_params, improved_score = _local_search_step(
                    cand, rng, tau=500, n_steps=80
                )
                if improved_score > best_score:
                    best_score = improved_score
                    best_params = improved_params.copy()
            except Exception:
                pass
    
    # Phase 5: Final ultra-high tau polish
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        try:
            res = minimize(
                _exact_objective,
                best_params,
                args=(8000,),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            pts = _params_to_pts(res.x)
            score = _compute_min_ratio(pts)
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Reconstruct final points
    if best_params is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    final_pts = _params_to_pts(best_params)
    
    return final_pts

# EVOLVE-BLOCK-END
