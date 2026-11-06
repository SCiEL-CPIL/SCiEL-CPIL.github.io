"""
Strategy: Global optimization using Differential Evolution (DE) on a reduced 
radial-angular parameter space. 

The regular octagon is a local maximum. To find the global maximum (which is 
known to be higher than 1.0 for n=8), we need a global optimizer. 
Differential Evolution is robust and effective for this type of problem.

We parameterize the 8 points by their polar coordinates (r_i, theta_i) relative 
to the center of mass. This reduces the search space and handles rotational 
symmetry more naturally than raw Cartesian coordinates. We normalize the 
configuration to have a diameter of 2 before computing the score to handle 
scale invariance.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from math import cos, sin, pi, exp, log, sqrt

def _score(points):
    """
    Compute Delta / n^n for 8 points, where Delta is the product of ordered 
    pair distances after scaling to diameter 2.
    """
    n = len(points)
    pts = np.array(points, dtype=float)
    
    # Check for distinctness
    if n < 2:
        return 0.0
        
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    # We want the max distance to be 2.
    # scaled_dists = dists * (2.0 / diam)
    
    # Product of ordered pair distances (exclude self-distances which are 0)
    # We need product over all ordered pairs (i,j) with i != j
    mask = ~np.eye(n, dtype=bool)
    # ordered_dists = scaled_dists[mask]
    
    # Delta = product of all ordered pair distances
    # log(Delta) = sum(log(d_ij)) for i!=j
    # Since we scale by 2/diam, log(d_scaled) = log(d) + log(2/diam)
    # There are n*(n-1) terms.
    # log_score = log(Delta_scaled) - n*log(n)
    # log(Delta_scaled) = sum_{i!=j} (log(d_ij) + log(2/diam))
    #                   = sum_{i!=j} log(d_ij) + n*(n-1)*log(2/diam)
    
    # To avoid underflow/overflow, work in log space
    # log_dists = np.log(dists[mask])
    # log_delta_scaled = np.sum(log_dists) + n*(n-1)*np.log(2.0/diam)
    # log_score = log_delta_scaled - n*np.log(n)
    
    # Let's compute directly on the scaled distances for clarity, 
    # but be careful with zeros.
    scaled_dists = dists / diam * 2.0
    ordered_dists = scaled_dists[mask]
    
    # Filter out any numerical zeros just in case (though distinctness is checked)
    # If any distance is 0, score is 0
    if np.any(ordered_dists < 1e-12):
        return 0.0
        
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def _params_to_points(params):
    """
    Convert parameters to 8 points.
    Parameters: [r0, t0, r1, t1, ..., r7, t7]
    r_i >= 0, t_i in [0, 2*pi)
    
    We center the points at the origin by subtracting the centroid.
    This removes translational invariance.
    """
    n = 8
    radii = params[:n]
    angles = params[n:]
    
    # Ensure radii are positive
    radii = np.abs(radii)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    pts = np.column_stack((x, y))
    
    # Center at origin to remove translation
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    
    return pts

def _neg_score_from_params(params):
    """
    Objective function for the optimizer. Returns negative score.
    """
    try:
        pts = _params_to_points(params)
        sc = _score(pts)
        return -sc
    except Exception:
        return 1e6 # Large penalty for invalid inputs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    best_score = 0.0
    best_points = None
    
    # 1. Initialize with regular octagon
    # Regular octagon points on unit circle
    regular_octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    current_score = _score(regular_octagon)
    if current_score > best_score:
        best_score = current_score
        best_points = regular_octagon.copy()
    
    # 2. Global Optimization using Differential Evolution
    # Parameter bounds:
    # Radii: [0, 10] (scale is handled by normalization in score, but DE needs finite bounds)
    # Angles: [0, 2*pi]
    bounds = []
    for i in range(8):
        bounds.append((0, 10))  # Radii
    for i in range(8):
        bounds.append((0, 2 * pi))  # Angles
        
    # DE is stochastic, so set seed for reproducibility if needed, but usually 
    # we want to explore. However, for deterministic evaluation, we might fix seed.
    # The task allows randomness, but deterministic is safer for "reproducibility" 
    # in some contexts. Here, we'll use a fixed seed to ensure the "best found" 
    # is consistent across runs if the evaluator runs it multiple times.
    # Actually, DE with a fixed seed is deterministic.
    
    try:
        # Use differential_evolution
        # maxiter: 200 generations
        # popsize: 15 (15 * 8 = 120 individuals per generation)
        # tol: 1e-6
        # Seed: 42
        
        result = differential_evolution(
            _neg_score_from_params, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8,
            polish=False  # We will do our own local polish
        )
        
        if result.success:
            best_params = result.x
            best_pts = _params_to_points(best_params)
            sc = _score(best_pts)
            if sc > best_score:
                best_score = sc
                best_points = best_pts
        else:
            # Even if not successful, check the best found
            best_params = result.x
            best_pts = _params_to_points(best_params)
            sc = _score(best_pts)
            if sc > best_score:
                best_score = sc
                best_points = best_pts
                
    except Exception as e:
        # If DE fails, fall back to best so far
        pass
    
    # 3. Local Refinement using Nelder-Mead on the best point found by DE
    # This helps to squeeze out the last bit of precision.
    if time.time() < deadline - 5.0 and best_points is not None:
        x0_flat = best_points.flatten()
        
        def _score_flat(x):
            pts = x.reshape(8, 2)
            return _score(pts)
        
        def _neg_score_flat(x):
            return -_score_flat(x)
        
        try:
            # Nelder-Mead on Cartesian coordinates
            # This is 16-dimensional, which is high, but NM can work if started close.
            result = minimize(_neg_score_flat, x0_flat, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if result.success:
                candidate_pts = result.x.reshape(8, 2)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts
        except Exception:
            pass

    # 4. Try a few random multi-starts with local optimization if time permits
    # This helps if DE got stuck in a local minimum.
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random start: perturb the best point significantly
            # Or generate a random configuration
            # Let's try random angles and radii
            radii = rng.uniform(0.5, 2.0, 8)
            angles = rng.uniform(0, 2*pi, 8)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            pts = np.column_stack((x, y))
            pts -= np.mean(pts, axis=0)
            
            x_start = pts.flatten()
            
            try:
                result = minimize(_neg_score_flat, x_start, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                
                if result.success:
                    candidate_pts = result.x.reshape(8, 2)
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_pts
            except Exception:
                continue

    # Final check: Ensure distinctness and bounds
    if best_points is None:
        best_points = regular_octagon
    
    # Check distinctness
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon if points collapsed
        best_points = regular_octagon
    
    # Ensure coordinates are within bounds
    # |coordinate| < 10^6 is easily satisfied
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
