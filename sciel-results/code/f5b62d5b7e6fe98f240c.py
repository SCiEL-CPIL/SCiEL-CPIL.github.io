"""
Strategy: Global Optimization via Differential Evolution on Polar Coordinates.
The previous approach used local Nelder-Mead on a parameterization constrained 
to be close to a regular octagon (radial/angular deformations). This approach 
uses Differential Evolution (DE) for global search over 8 radii and 8 angles 
(polar coordinates). This allows the optimizer to explore a much broader space 
of configurations, including those where points are not ordered by angle or 
where radii vary significantly, which may be necessary to reach the global 
maximum for n=8.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def _score(points):
    """
    Compute Delta / n^n for 8 points.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    """
    n = len(points)
    pts = np.array(points, dtype=float)
    
    # Ensure points are 2D
    if pts.shape[1] != 2:
        return 0.0
        
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    # We only care about the product of non-zero distances
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = dists[mask]
    
    # Scale distances
    scaled_dists = ordered_dists / diam * 2.0
    
    # Check for distinct points (distance > 0)
    if np.any(scaled_dists < 1e-12):
        return 0.0
        
    # Product of ordered pair distances
    # log(Delta) = sum(log(scaled_dists))
    log_delta = np.sum(np.log(scaled_dists))
    
    # Score = Delta / n^n
    # log(Score) = log_delta - n*log(n)
    log_score = log_delta - n * np.log(n)
    
    return np.exp(log_score)

def _params_to_points(params):
    """
    Convert 16 parameters (8 r, 8 theta) to 8 (x,y) points.
    params[0:8] are radii, params[8:16] are angles.
    """
    n = 8
    r = params[:n]
    theta = params[n:]
    
    # Ensure radii are non-negative (DE might go negative, clip to 0)
    r = np.maximum(r, 0.0)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack((x, y))
    return points

def _neg_score_from_params(params):
    """
    Objective function for optimization.
    params: 16 floats (r0...r7, theta0...theta7)
    """
    pts = _params_to_points(params)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 8
    # Bounds for r and theta
    # r: 0 to 10 (scaling is invariant, so absolute scale doesn't matter, 
    #    but we need a range to work with. Diameter will be normalized.)
    # theta: 0 to 2*pi
    bounds = [(0.0, 10.0)] * n + [(0.0, 2.0 * np.pi)] * n
    
    # Differential Evolution
    # We use a high maxiter and a reasonable popsize to allow global search.
    # 'polish' is True by default, which uses L-BFGS-B to refine the best result.
    # We disable polish initially to save time, and do a manual polish later 
    # if time permits, or just rely on DE's high precision.
    # Actually, polish is good. Let's keep it but control maxiter.
    
    try:
        # Differential Evolution
        # popsize: 15 (default is 15)
        # maxiter: 500 (default is 1000, might be too slow. 500 should be enough for DE to converge roughly)
        # tol: 1e-8
        # strategy: 'best1bin' (default)
        # maxfevals: None (controlled by maxiter * popsize * 10 roughly)
        
        res = differential_evolution(
            _neg_score_from_params, 
            bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.4, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol' # Use Sobol for better initial distribution
        )
        
        best_params = res.x
        best_score = -res.fun
        
    except Exception:
        # Fallback to regular octagon if DE fails
        # r = 1, theta = k*pi/4
        r = np.ones(8)
        theta = np.arange(8) * (np.pi / 4)
        best_params = np.concatenate([r, theta])
        best_score = _score(_params_to_points(best_params))

    # If we have time, do a second phase: local refinement on the best DE result
    # using a more precise method or just running DE again with tighter bounds around the best.
    if time.time() < deadline - 10.0:
        # We can try to refine the best point found by DE.
        # Since DE with polish already does this, we might just check if we can squeeze more.
        # Let's try a local Nelder-Mead on the best params found by DE.
        from scipy.optimize import minimize
        
        try:
            res_nm = minimize(
                _neg_score_from_params, 
                best_params, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            
            nm_score = -res_nm.fun
            if nm_score > best_score:
                best_score = nm_score
                best_params = res_nm.x
        except Exception:
            pass

    # Convert best params to points
    points = _params_to_points(best_params)
    
    # Sanity check: ensure distinct points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        # Fallback to regular octagon if points collapsed
        octagon = np.array([[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)] for k in range(8)])
        points = octagon
    
    # Ensure coordinates are within bounds (they should be, but just in case)
    # The problem requires |coord| < 1e6. Our search was bounded, so this is safe.
    
    return [[float(p[0]), float(p[1])] for p in points]
# EVOLVE-BLOCK-END
