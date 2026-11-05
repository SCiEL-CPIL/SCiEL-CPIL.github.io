"""Optimized program for 8-point max product of distances at fixed diameter."""
import time
import numpy as np
from scipy.optimize import minimize

def _score_from_points(points):
    """Compute Delta / n^n where Delta is product of ordered pair distances, scaled to diameter 2."""
    points = np.array(points, dtype=float)
    n = len(points)
    # Compute all pairwise distances
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return -np.inf
    # Scale to diameter 2
    dists_scaled = dists / diam * 2.0
    # Product over ordered pairs (i != j)
    # Use log for numerical stability
    mask = ~np.eye(n, dtype=bool)
    log_dists = np.log(dists_scaled[mask])
    log_delta = np.sum(log_dists)
    score = np.exp(log_delta) / (n ** n)
    return score

def _generate_points_from_params(params):
    """Generate 8 points from 3 parameters: r1, r2, theta_offset.
    
    Square 1 (even indices): radius r1, angles 0, pi/2, pi, 3pi/2
    Square 2 (odd indices): radius r2, angles pi/4, 3pi/4, 5pi/4, 7pi/4 + theta_offset
    """
    r1, r2, theta = params
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]) + theta
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    # Interleave: point 0 from sq1, point 1 from sq2, etc.
    points = np.zeros((8, 2))
    points[0::2] = pts1
    points[1::2] = pts2
    return points

def _negative_score(params):
    points = _generate_points_from_params(params)
    s = _score_from_points(points)
    if not np.isfinite(s):
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Regular octagon as baseline
    baseline_points = [[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)]
    best_score = _score_from_points(baseline_points)
    best_points = baseline_points
    
    # Parameterization: r1, r2, theta
    # Regular octagon corresponds to r1 = r2 = 1, theta = 0
    # Try multiple starting points for L-BFGS-B
    
    initial_guesses = [
        [1.0, 1.0, 0.0],          # Regular octagon
        [1.0, 1.0, 0.1],          # Slight rotation
        [1.0, 1.0, -0.1],
        [0.9, 1.1, 0.0],          # Different radii
        [1.1, 0.9, 0.0],
        [0.8, 1.2, 0.2],
        [1.2, 0.8, -0.2],
        [0.95, 1.05, 0.15],
        [1.05, 0.95, -0.15],
        [0.85, 1.15, 0.3],
        [1.15, 0.85, -0.3],
        [0.7, 1.3, 0.4],
        [1.3, 0.7, -0.4],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _negative_score,
                guess,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 0:
                s = -result.fun
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = _generate_points_from_params(result.x)
        except Exception:
            continue
    
    # Also try a 4-parameter version: allow the two squares to have different center offsets
    # Or try a more general parameterization
    
    # Try grid search + refinement for better coverage
    if time.time() < deadline - 30:
        # Coarse grid over r1, r2, theta
        r1_vals = np.linspace(0.5, 1.5, 15)
        r2_vals = np.linspace(0.5, 1.5, 15)
        theta_vals = np.linspace(-0.5, 0.5, 11)
        
        grid_best_score = best_score
        grid_best_params = None
        
        for r1 in r1_vals:
            for r2 in r2_vals:
                for theta in theta_vals:
                    if time.time() > deadline - 30:
                        break
                    params = [r1, r2, theta]
                    s = -_negative_score(params)
                    if np.isfinite(s) and s > grid_best_score:
                        grid_best_score = s
                        grid_best_params = params
        
        if grid_best_params is not None and grid_best_score > best_score:
            # Refine from best grid point
            try:
                result = minimize(
                    _negative_score,
                    grid_best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                s = -result.fun
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = _generate_points_from_params(result.x)
            except Exception:
                pass
    
    # Final polish: high-precision local optimization
    if time.time() < deadline - 5 and best_points is not None:
        # Convert current best back to approximate parameters for polishing
        # Use the best params from optimization if available, otherwise try to recover
        # For now, do a local Nelder-Mead on the full 16D space around best_points
        try:
            def neg_score_full(pts_flat):
                pts = pts_flat.reshape(8, 2)
                s = _score_from_points(pts)
                if not np.isfinite(s):
                    return 1e10
                return -s
            
            initial_flat = np.array(best_points).flatten()
            result = minimize(
                neg_score_full,
                initial_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            s = -result.fun
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_points = result.x.reshape(8, 2).tolist()
        except Exception:
            pass
    
    # Ensure distinct points and valid coordinates
    result_points = []
    seen = set()
    for p in best_points:
        x, y = float(p[0]), float(p[1])
        key = (round(x, 12), round(y, 12))
        if key not in seen:
            seen.add(key)
            result_points.append([x, y])
    
    # If we lost points due to deduplication, add small perturbations
    while len(result_points) < 8:
        result_points.append([0.001 * (len(result_points) - 7), 0.001 * (7 - len(result_points))])
    
    return result_points[:8]
