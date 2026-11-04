"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_symmetric_points(r_inner, theta_offset):
    """
    Generate 14 points: 7 on unit circle, 7 on circle of radius r_inner.
    Used as a high-quality initial guess for the full-dimensional search.
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta_offset
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _pattern_search(pts0, max_iter=200, initial_step=0.05, min_step=1e-4, time_limit=10.0):
    """
    Perform a direct pattern search to maximize the score.
    This breaks symmetry by allowing all 28 coordinates to move.
    """
    start_time = time.time()
    current_pts = pts0.copy()
    current_score = _score(current_pts)
    step = initial_step
    n = len(current_pts)
    
    # We optimize in 2D space. 
    # To ensure we don't just drift the whole shape, we could normalize, 
    # but the score is scale-invariant (area/area), so drift is harmless 
    # unless it causes numerical issues. However, keeping points in a 
    # reasonable range is good.
    
    while step > min_step:
        improved = False
        # Check all 28 coordinates, both + and - directions
        for i in range(n):
            for coord in range(2):
                if time.time() > start_time + time_limit:
                    return current_pts, current_score
                
                # Try positive step
                trial_pts = current_pts.copy()
                trial_pts[i, coord] += step
                trial_score = _score(trial_pts)
                
                if trial_score > current_score + 1e-12:
                    current_pts = trial_pts
                    current_score = trial_score
                    improved = True
                    continue # Restart loop to check other coords with new center
                    
                # Try negative step
                trial_pts = current_pts.copy()
                trial_pts[i, coord] -= step
                trial_score = _score(trial_pts)
                
                if trial_score > current_score + 1e-12:
                    current_pts = trial_pts
                    current_score = trial_score
                    improved = True
                    continue
        
        if not improved:
            step *= 0.5
        else:
            # If we improved, we might keep the same step to explore further
            # in that direction, but standard pattern search often keeps step
            # or slightly increases. Here we keep it simple: keep step if improved,
            # but to avoid oscillation, we could also just proceed.
            # Actually, if we improved, we should continue with the same step size
            # to see if further moves in the same direction are better.
            # But the loop above restarts after an improvement, so it effectively
            # does a greedy hill climb.
            pass
            
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 35s safety margin
    
    best_score = -float('inf')
    best_pts = None
    
    # 1. Find a good symmetric starting point using the previous 2D search logic
    # but faster. We know r~0.5-0.8 and theta~pi/7 is good.
    # Let's do a quick coarse grid to find the best symmetric seed.
    
    r_range = np.linspace(0.4, 0.8, 10)
    theta_range = np.linspace(0, 2*np.pi/7, 10, endpoint=False)
    
    coarse_best_score = -float('inf')
    coarse_best_pts = None
    
    for r in r_range:
        for theta in theta_range:
            if time.time() > deadline - 30.0: # Leave time for pattern search
                break
            pts = _generate_symmetric_points(r, theta)
            s = _score(pts)
            if s > coarse_best_score:
                coarse_best_score = s
                coarse_best_pts = pts
        
        if time.time() > deadline - 30.0:
            break
            
    if coarse_best_pts is not None:
        best_pts = coarse_best_pts.copy()
        best_score = coarse_best_score
    else:
        # Fallback
        best_pts = _generate_symmetric_points(0.6, np.pi/7)
        best_score = _score(best_pts)

    # 2. Run Pattern Search (Full 28D) from the best symmetric seed
    # This allows symmetry breaking.
    
    if time.time() < deadline - 10.0:
        # Run a few pattern searches from slightly perturbed versions of the best seed
        # to avoid getting stuck in a specific local optimum of the 28D landscape.
        
        rng = np.random.default_rng(42)
        
        for i in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the seed slightly
            noise = rng.normal(0, 0.05, size=best_pts.shape)
            init_pts = best_pts + noise
            
            # Ensure it's not degenerate
            if _score(init_pts) <= 0:
                continue
                
            try:
                # Limit time per search to ensure we finish
                remaining = deadline - 5.0 - time.time()
                if remaining < 1.0:
                    break
                
                # Pattern search is iterative. We give it a chunk of time.
                # We'll run it for a fixed number of iterations or until time runs out.
                # To be safe, we run a single pattern search with a tight time limit per call.
                res_pts, res_score = _pattern_search(
                    init_pts, 
                    max_iter=100, 
                    initial_step=0.1, 
                    min_step=1e-5, 
                    time_limit=min(20.0, remaining)
                )
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts
            except Exception:
                pass

    # 3. Final Polish with Nelder-Mead on the best point found
    # This can often squeeze out a bit more than pattern search.
    
    if time.time() < deadline - 5.0:
        def _negative_score_full(params):
            pts = params.reshape(14, 2)
            s = _score(pts)
            if s < 0:
                return 1e9
            return -s
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(_negative_score_full, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500, 'maxfev': 500})
            
            if res.fun < -best_score:
                final_pts = res.x.reshape(14, 2)
                final_score = _score(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_pts = final_pts
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
