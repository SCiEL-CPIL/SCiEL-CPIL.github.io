"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    # We need product over all ordered pairs (i,j) with i != j
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_score(points):
    return -_score(points)

def _params_to_points(params):
    """Convert (r0..r7, a0..a7) to 8 points.
    To ensure distinctness and avoid degeneracy, we assume a general star-polygon-like ordering 
    but allow radii and angles to vary freely. 
    However, to maintain the 'almost regular' structure which is likely optimal, 
    we can parameterize as deviations from a regular octagon.
    Let r_i = 1 + dr_i, a_i = base_a_i + da_i.
    """
    n = 8
    radii = params[:n]
    angles = params[n:]
    # Ensure angles are in a reasonable range to avoid wrap-around issues in optimization
    # The base angles are 2*pi*k/8
    base_angles = 2 * pi * np.arange(n) / n
    angles = base_angles + angles
    
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    return np.column_stack([xs, ys])

def _score_from_params(params):
    pts = _params_to_points(params)
    return _score(pts)

def _neg_score_from_params(params):
    return -_score_from_params(params)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with regular octagon
    # Parameters: 8 radii (deviation from 1.0) and 8 angles (deviation from base)
    # Initial params are all zeros for regular octagon
    n = 8
    x0 = np.zeros(2 * n)
    
    best_params = x0.copy()
    best_score = _score_from_params(best_params)
    
    # Strategy: Direct optimization on the 16 intrinsic parameters (radii, angles)
    # This is a lower-dimensional, more structured space than Cartesian coordinates.
    # We use L-BFGS-B with numerical gradients for robustness, or Nelder-Mead.
    # Since the landscape might be smooth near the regular octagon, L-BFGS-B with
    # finite differences should work well. However, Nelder-Mead is derivative-free
    # and often more robust for these geometric problems. Let's use a combination.
    
    rng = np.random.RandomState(42)
    
    # 1. Try a local refinement from the regular octagon
    try:
        result = minimize(_neg_score_from_params, x0, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        
        if time.time() < deadline - 10.0:
            candidate_params = result.x
            candidate_score = _score_from_params(candidate_params)
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params
    except Exception:
        pass

    # 2. Try multi-start with perturbations in the parameter space
    # The optimal configuration for n=8 might not be the regular octagon.
    # It could be a configuration with some symmetry breaking.
    # We try perturbations of the angles and radii.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random perturbation
        # Small perturbations first, then larger
        scale = 0.01 * (1 + i / 10.0)
        perturbation = rng.randn(2 * n) * scale
        
        x_start = best_params + perturbation
        
        try:
            result = minimize(_neg_score_from_params, x_start, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            candidate_params = result.x
            candidate_score = _score_from_params(candidate_params)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params
        except Exception:
            continue

    # 3. Try a specific known configuration: The "almost regular" octagon
    # Sometimes optimal points for max product of distances are vertices of a regular polygon
    # but with some points moved slightly.
    
    # Let's also try a configuration where points are on two circles?
    # Or maybe just stick to the radial/angular parameterization which covers a wide range.
    
    # Final check: Convert best_params to points
    best_points = _params_to_points(best_params)
    
    # Ensure distinctness
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon if points collapsed
        octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        best_points = np.array(octagon, dtype=float)
    
    # Ensure coordinates are within bounds
    # |coordinate| < 10^6 is easily satisfied since we start near 1.0
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
