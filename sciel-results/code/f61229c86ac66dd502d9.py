"""Packing 50 equal circles in a unit circle.
Strategy:
1. Parametric Global Search: Model packing as concentric rings (1, 6, 12, 18, 13).
   Optimize ring radii and rotations using Differential Evolution (8D).
2. Local Polish: Refine all 100 coordinates using L-BFGS-B with a smooth soft-min objective.
3. Final Polish: Higher precision refinement.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist

N = 50
TIME_LIMIT = 145.0  # Safety margin before 150s kill

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
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    flat_vals = pair_vals.flatten()
    flat_vals = flat_vals[~np.isinf(flat_vals)]
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([flat_vals, boundary_vals])
    
    # Soft-min via log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _smooth_objective(x, beta=100.0):
    """
    Objective for L-BFGS-B: minimize negative soft-min radius + penalty.
    """
    pts = x.reshape(N, 2)
    
    # Hard penalty for points outside unit circle
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    
    # Negative soft-min radius
    radius = _soft_min_surrogate(pts, beta)
    
    return -radius + penalty

def _polish_with_smooth(points, deadline, betas=None, maxiter=200):
    """
    Polish points using L-BFGS-B with increasing beta.
    """
    if betas is None:
        betas = [50, 100, 200, 500, 1000]
    
    current_points = points.copy()
    current_score = _compute_score(current_points)
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        def obj(x):
            return _smooth_objective(x, beta)
        
        try:
            result = minimize(
                obj,
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_points = result.x.reshape(N, 2)
            
            # Check validity
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                # Clip points that went slightly outside due to numerical error
                if np.max(norms) > 1.0:
                    mask = norms > 1.0
                    new_points[mask] /= norms[mask, np.newaxis] * 1.0001
                    
                new_score = _compute_score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
        except Exception:
            pass
    
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # --- Phase 1: Parametric Global Search ---
    
    # Bounds for parameters:
    # Radii: 0.1 to 0.95
    # Angles: 0 to 2*pi
    bounds = [(0.1, 0.95)] * 4 + [(0.0, 2 * np.pi)] * 4
    
    best_param_points = None
    best_score = -np.inf
    
    # Run Differential Evolution
    # We limit the time for this phase to ensure we have time for polishing
    de_time_budget = 60.0
    
    try:
        # Use a smaller population and maxiter to fit in time budget
        # We can also use a wrapper to stop early if time runs out
        def wrapper(params):
            if time.time() > start_time + de_time_budget:
                raise TimeoutError
            return _parametric_objective(params)
        
        result = differential_evolution(
            wrapper,
            bounds,
            seed=42,
            maxiter=50, # Keep low, DE is slow per iter
            popsize=10,
            tol=1e-6,
            mutation=(0.3, 1.2),
            recombination=0.8
        )
        
        if result.x is not None:
            best_param_points = _build_points_from_params(result.x)
            best_score = _compute_score(best_param_points)
            
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Fallback if DE failed or timed out immediately: use a reasonable static initialization
    if best_score < 0.1:
        # Static heuristic initialization
        # Radii approx: 0.25, 0.5, 0.75, 0.9
        # Angles: 0, 0, 0, 0
        init_params = np.array([0.25, 0.5, 0.75, 0.9, 0.0, 0.0, 0.0, 0.0])
        best_param_points = _build_points_from_params(init_params)
        best_score = _compute_score(best_param_points)
        
        # Also try a few random offsets
        rng = np.random.default_rng(42)
        for _ in range(10):
            params = np.array([
                0.25 + rng.normal(0, 0.05),
                0.50 + rng.normal(0, 0.05),
                0.75 + rng.normal(0, 0.05),
                0.90 + rng.normal(0, 0.05),
                rng.uniform(0, 2*np.pi),
                rng.uniform(0, 2*np.pi),
                rng.uniform(0, 2*np.pi),
                rng.uniform(0, 2*np.pi)
            ])
            pts = _build_points_from_params(params)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_param_points = pts

    current_points = best_param_points.copy()
    current_score = best_score
    
    # --- Phase 2: Local Polish with Smooth Surrogate ---
    
    if time.time() < deadline - 10:
        # Use L-BFGS-B with increasing beta to sharpen the objective
        polished = _polish_with_smooth(current_points, deadline, betas=[50, 100, 200, 500, 1000], maxiter=200)
        polished_score = _compute_score(polished)
        if polished_score > current_score:
            current_points = polished
            current_score = polished_score
    
    # --- Phase 3: Final High-Precision Polish ---
    
    if time.time() < deadline - 2:
        # Use higher beta and more iterations for final refinement
        polished = _polish_with_smooth(current_points, deadline, betas=[1000, 2000, 5000], maxiter=300)
        polished_score = _compute_score(polished)
        if polished_score > current_score:
            current_points = polished
            current_score = polished_score
    
    # Ensure valid output
    current_points = np.array(current_points, dtype=np.float64)
    norms = np.linalg.norm(current_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        current_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return current_points
