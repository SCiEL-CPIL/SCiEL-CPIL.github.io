"""
Pack 50 equal circles in the unit disk.
Strategy:
1. Multi-start initialization: parametric ring model + hexagonal lattice.
2. Parametric Search: Optimize ring radii and rotations using Differential Evolution.
3. Local Polish: Refine all 100 coordinates using L-BFGS-B with smooth soft-min objective.
4. Exact Polish: Final high-precision nudge on the true nonsmooth objective.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

N = 50
TIME_LIMIT = 142.0  # Safety margin before 150s kill

# Ring structure: 1 center, 6, 12, 18, 13
RING_SIZES = [1, 6, 12, 18, 13]
N_RINGS = len(RING_SIZES)

def _compute_score(points):
    """Compute the maximum radius for given points."""
    if not np.all(np.isfinite(points)):
        return 0.0
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _build_points_from_params(params):
    """
    Construct 50 points from parametric representation.
    params: array of shape (8,).
    First 4 values: radii for rings 1-4 (center is always 0).
    Next 4 values: rotation angles for rings 1-4.
    """
    points = np.zeros((N, 2))
    idx = 0
    
    # Ring 0: Center
    points[idx] = [0.0, 0.0]
    idx += 1
    
    for i in range(1, N_RINGS):
        count = RING_SIZES[i]
        radius = params[i-1]
        angle_offset = params[N_RINGS - 1 + (i-1)]
        
        for j in range(count):
            angle = 2 * np.pi * j / count + angle_offset
            points[idx, 0] = radius * np.cos(angle)
            points[idx, 1] = radius * np.sin(angle)
            idx += 1
            
    return points

def _parametric_objective(params):
    """
    Objective for differential evolution: minimize negative score.
    """
    points = _build_points_from_params(params)
    score = _compute_score(points)
    return -score

def _soft_min_surrogate(points, beta=100.0):
    """
    Compute a smooth surrogate for the min of:
    - pairwise distances / 2
    - boundary distances (1 - norm)
    Uses numerically stable log-sum-exp.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances (upper triangle only to save memory/time)
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    # Only use upper triangle to avoid double-counting
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft-min via numerically stable log-sum-exp
    m = np.min(all_vals)
    # Clip to prevent overflow in exp
    shifted = all_vals - m
    shifted = np.clip(shifted, -50.0, 50.0)  # exp(-50) ~ 2e-22, safe
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    if sum_exp > 0:
        log_sum = np.log(sum_exp)
    else:
        log_sum = 0.0
    soft_min = m - log_sum / beta
    
    return soft_min

def _smooth_objective(x, beta=100.0):
    """
    Objective for L-BFGS-B: minimize negative soft-min radius + penalty.
    """
    pts = x.reshape(N, 2)
    
    # Hard penalty for points outside unit circle (quadratic for smoothness)
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0.0, norms - 1.0)
    penalty = 500.0 * np.sum(boundary_violation ** 2)
    
    # Negative soft-min radius
    radius = _soft_min_surrogate(pts, beta)
    
    return -radius + penalty

def _hexagonal_initial(ring_radii=None):
    """
    Generate a hexagonal lattice-based initialization.
    Places circles in a hexagonal pattern, then scales to fit.
    """
    if ring_radii is None:
        ring_radii = [0.0, 0.25, 0.50, 0.75, 0.92]
    
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # Hexagonal rings
    for ring_idx in range(1, 5):
        radius = ring_radii[ring_idx]
        count = 6 * ring_idx
        for j in range(count):
            angle = 2 * np.pi * j / count
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points[:N])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # --- Phase 1: Multi-start Initialization ---
    
    candidates = []
    
    # Candidate 1: Parametric ring search via DE
    bounds = [(0.1, 0.95)] * 4 + [(0.0, 2 * np.pi)] * 4
    de_time_budget = 30.0  # Reduced from 60s to leave more time for polishing
    
    try:
        def wrapper(params):
            if time.time() > start_time + de_time_budget:
                raise TimeoutError
            return _parametric_objective(params)
        
        result = differential_evolution(
            wrapper,
            bounds,
            seed=42,
            maxiter=30,
            popsize=8,
            tol=1e-5,
            mutation=(0.3, 1.2),
            recombination=0.8
        )
        
        if result.x is not None:
            pts = _build_points_from_params(result.x)
            s = _compute_score(pts)
            candidates.append((s, pts))
            
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Candidate 2: Static ring initialization with random perturbations
    rng = np.random.default_rng(42)
    for trial in range(20):
        params = np.array([
            0.25 + rng.normal(0, 0.03),
            0.50 + rng.normal(0, 0.03),
            0.75 + rng.normal(0, 0.03),
            0.92 + rng.normal(0, 0.02),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi)
        ])
        pts = _build_points_from_params(params)
        s = _compute_score(pts)
        candidates.append((s, pts))
    
    # Candidate 3: Hexagonal lattice initialization
    hex_pts = _hexagonal_initial()
    hex_score = _compute_score(hex_pts)
    candidates.append((hex_score, hex_pts))
    
    # Select best candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    current_points = candidates[0][1].copy()
    current_score = candidates[0][0]
    
    # --- Phase 2: Local Polish with Smooth Surrogate ---
    
    if time.time() < deadline - 15:
        # Use L-BFGS-B with increasing beta to sharpen the objective
        betas = [30, 80, 200, 500]
        
        for beta in betas:
            if time.time() > deadline - 8:
                break
                
            def obj(x):
                return _smooth_objective(x, beta)
            
            try:
                result = minimize(
                    obj,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
                )
                
                new_points = result.x.reshape(N, 2)
                
                # Check validity
                if np.all(np.isfinite(new_points)):
                    norms = np.linalg.norm(new_points, axis=1)
                    # Clip points that went slightly outside due to numerical error
                    if np.max(norms) > 1.0:
                        mask = norms > 1.0
                        new_points[mask] /= norms[mask, np.newaxis] * 1.00001
                        
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current_points = new_points.copy()
                        current_score = new_score
            except Exception:
                pass

    # --- Phase 3: Exact Polish with Proper Backtracking ---
    
    if time.time() < deadline - 3:
        best_pts = current_points.copy()
        best_r = current_score
        
        for iteration in range(100):
            if time.time() > deadline - 2:
                break
            
            pts = best_pts.copy()
            
            # Find the most violated constraint
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_pair_idx
            min_dist = dists[i, j]
            min_pair_r = min_dist / 2.0
            
            norms = np.linalg.norm(pts, axis=1)
            min_boundary_idx = np.argmax(norms)
            min_boundary = 1.0 - norms[min_boundary_idx]
            
            r = min(min_pair_r, min_boundary)
            
            # Try several step sizes
            improved = False
            for step in [0.005, 0.002, 0.001, 0.0005]:
                trial_pts = pts.copy()
                
                if min_pair_r <= min_boundary:
                    if min_dist > 1e-10:
                        dir = (trial_pts[i] - trial_pts[j]) / min_dist
                        trial_pts[i] += dir * step
                        trial_pts[j] -= dir * step
                else:
                    if norms[min_boundary_idx] > 1e-10:
                        dir = trial_pts[min_boundary_idx] / norms[min_boundary_idx]
                        trial_pts[min_boundary_idx] -= dir * step
                
                # Project back to disk if necessary
                trial_norms = np.linalg.norm(trial_pts, axis=1)
                mask = trial_norms > 1.0
                if np.any(mask):
                    trial_pts[mask] /= trial_norms[mask, np.newaxis] * 1.00001
                
                trial_score = _compute_score(trial_pts)
                if trial_score > best_r + 1e-12:
                    best_pts = trial_pts.copy()
                    best_r = trial_score
                    improved = True
                    break
            
            if not improved:
                break
        
        if best_r > current_score:
            current_points = best_pts
            current_score = best_r

    return current_points
