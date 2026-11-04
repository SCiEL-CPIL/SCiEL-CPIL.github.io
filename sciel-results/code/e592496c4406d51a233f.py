"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices for speed
_triple_indices = np.array(list(combinations(range(N), 3)))

# EVOLVE-BLOCK-START

# Number of Gaussian basis functions for radial profile
NUM_BASIS = 5

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
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-9:
        return 0.0
        
    return min_area / hull_area

def _radial_profile(params, angles):
    """
    Compute radius as a function of angle using Gaussian basis expansion.
    params structure: [base_radius, (center_1, width_1, amp_1), (center_2, width_2, amp_2), ...]
    Total params: 1 + 3 * NUM_BASIS
    """
    base_r = params[0]
    radii = np.full_like(angles, base_r)
    
    for i in range(NUM_BASIS):
        center = params[1 + 3*i]
        width = params[2 + 3*i]
        amp = params[3 + 3*i]
        
        # Circular distance
        diff = angles - center
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # Gaussian kernel
        g = amp * np.exp(-0.5 * (diff / width) ** 2)
        radii += g
    
    return radii

def _params_to_points(params):
    """Convert basis parameters to point coordinates."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = _radial_profile(params, angles)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.05)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _smooth_min(ratios, tau):
    """Smooth approximation of minimum using log-sum-exp."""
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -700, 0)  # Prevent overflow
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_r - (1.0 / tau) * log_sum

def _objective(params, tau):
    """
    Objective function: minimize negative smoothed min ratio.
    params: [base_radius, (center_1, width_1, amp_1), ...]
    """
    pts = _params_to_points(params)
    
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e9
    
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-9:
        return 1e9
    
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    approx_min = _smooth_min(ratios, tau)
    return -approx_min

def _de_objective(params, tau):
    """Wrapper for differential evolution that handles edge cases."""
    try:
        val = _objective(params, tau)
        if np.isnan(val) or np.isinf(val) or val > 1e8:
            return 1e8
        return val
    except Exception:
        return 1e8

def _refine_with_lbfgs(params, tau, time_budget=5.0):
    """Refine parameters using L-BFGS-B for local optimization."""
    start = time.time()
    
    n_params = len(params)
    # Reasonable bounds
    bounds = [(0.3, 3.0)]  # base_radius
    for _ in range(NUM_BASIS):
        bounds.append((-np.pi, np.pi))  # center
        bounds.append((0.1, 1.5))       # width
        bounds.append((-0.5, 0.5))      # amplitude
    
    def obj(params):
        if time.time() - start > time_budget:
            return 1e9
        try:
            return _objective(params, tau)
        except Exception:
            return 1e9
    
    try:
        res = minimize(
            obj,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < 1e8:
            return res.x
    except Exception:
        pass
    
    return params

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Total number of parameters: 1 (base) + 3 * NUM_BASIS
    n_params = 1 + 3 * NUM_BASIS
    
    # Bounds for differential evolution
    bounds = [(0.5, 2.0)]  # base_radius
    for _ in range(NUM_BASIS):
        bounds.append((-np.pi, np.pi))  # center
        bounds.append((0.2, 1.2))       # width
        bounds.append((-0.4, 0.4))      # amplitude
    
    best_score = -np.inf
    best_params = None
    
    # Annealing schedule for tau
    taus = [50, 200, 1000]
    
    # Run differential evolution for each tau
    for tau in taus:
        if (time.time() - start_time) > time_limit - 30.0:
            break
        
        # Check if we have enough time for DE
        remaining = time_limit - 10.0 - (time.time() - start_time)
        if remaining < 15.0:
            break
        
        try:
            # Use a subset of the remaining time for DE
            maxiter = int(min(100, remaining / 0.5))  # ~0.5s per iteration estimate
            maxiter = max(maxiter, 20)
            
            result = differential_evolution(
                _de_objective,
                bounds,
                args=(tau,),
                seed=42,
                maxiter=maxiter,
                tol=1e-8,
                polish=False,  # We'll do our own polish
                update_interval=1,
                workers=-1,
                init='sobol'
            )
            
            if result.fun < 1e8:
                params = result.x
                # Refine with L-BFGS-B
                params = _refine_with_lbfgs(params, tau, time_budget=5.0)
                
                pts = _params_to_points(params)
                score = _compute_min_ratio(pts)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
        except Exception as e:
            pass
    
    # If DE didn't find anything good, try a simpler approach
    if best_params is None or best_score < 30.0:
        # Fall back to regular polygon with small perturbations
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        for restart in range(5):
            if (time.time() - start_time) > time_limit - 5.0:
                break
            
            # Create params: base radius 1, small random perturbations
            params = np.zeros(n_params)
            params[0] = 1.0  # base radius
            for i in range(NUM_BASIS):
                params[1 + 3*i] = rng.uniform(-np.pi, np.pi)  # random centers
                params[2 + 3*i] = rng.uniform(0.3, 1.0)       # moderate widths
                params[3 + 3*i] = rng.uniform(-0.15, 0.15)    # small amplitudes
            
            # Refine
            params = _refine_with_lbfgs(params, 100, time_budget=3.0)
            
            pts = _params_to_points(params)
            score = _compute_min_ratio(pts)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
    
    # Final refinement with high tau
    if best_params is not None and (time.time() - start_time) < time_limit - 5.0:
        best_params = _refine_with_lbfgs(best_params, 5000, time_budget=5.0)
        pts = _params_to_points(best_params)
        score = _compute_min_ratio(pts)
        if score > best_score:
            best_score = score
    
    # If still no good solution, return regular polygon
    if best_params is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
    
    final_pts = _params_to_points(best_params)
    return final_pts

# EVOLVE-BLOCK-END
