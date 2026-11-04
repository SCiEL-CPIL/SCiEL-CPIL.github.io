"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

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
    # We iterate over pairs (i, j) and vectorize over k
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

def _objective(x):
    """
    Objective function for minimization (we want to maximize score, so minimize -score).
    x is a flat array of shape (2*N,).
    """
    pts = x.reshape((N, 2))
    
    # Optional: Normalize to prevent scale drift issues in DE, though DE handles bounds.
    # However, the score is scale-invariant. 
    # To help DE, we can normalize the points to have unit RMS radius before scoring?
    # No, DE explores the space. If we normalize inside the objective, the landscape changes.
    # Better to let DE explore a bounded box, e.g., [-1, 1] x [-1, 1].
    # The score function itself is scale-invariant, so scaling doesn't change the value,
    # but it might affect the "gradient" perceived by DE if we were using gradients.
    # DE is derivative-free, so it just samples.
    
    s = _score(pts)
    return -s

def _normalize_radius(pts, target_rms=1.0):
    """
    Normalize the point set so that the RMS distance from the centroid is target_rms.
    """
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if rms < 1e-12:
        return pts
    scale = target_rms / rms
    return centered * scale + centroid

def _initial_construction_regular():
    """14 points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Define bounds for Differential Evolution
    # We search in a box [-1, 1] x [-1, 1]
    # Since the problem is scale invariant, the absolute scale doesn't matter,
    # but the relative positions do. A box of [-1, 1] is reasonable.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # DE Parameters
    # popsize: number of individuals in the population. Default 15.
    # mutation: (0.5, 1.0) is default.
    # recombination: 0.7 is default.
    # tol: 1e-8 is default.
    # maxiter: We need to control this via time.
    
    # Strategy: Run DE with a time limit.
    # scipy's differential_evolution doesn't have a direct time limit, 
    # but we can estimate the number of iterations or use a callback.
    # A callback is better.
    
    best_result = None
    best_score_val = -float('inf')
    
    def callback(xk, convergence):
        nonlocal best_result, best_score_val
        if time.time() > deadline:
            # Raise an exception to stop DE? 
            # Or just set a flag. DE checks for exceptions in the objective?
            # Actually, raising an exception in the objective kills the run.
            # It's safer to let it finish the current generation and check time in the loop if we wrote our own,
            # but with scipy DE, we can pass a maxiter that is likely to be hit before timeout
            # or just rely on the fact that DE is fast enough?
            # 28 dimensions, popsize 15, 100 iterations is 1500 evaluations.
            # Each evaluation is ~0.1ms? No, _score is O(N^3) with N=14.
            # 14 choose 3 = 364 triangles. ConvexHull is O(N log N).
            # It's very fast. 10000 evaluations should be well under 1 second.
            # So we can afford many iterations.
            pass

    # Let's try a robust setup.
    # We will run DE.
    try:
        # initial='sobol' is good for uniform coverage.
        # But DE with random init is fine.
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=200, # 200 iterations * popsize 15 = 3000 evals. Should be fast.
            popsize=20,
            tol=1e-8,
            init='sobol'
        )
        
        if result.success or result.fun is not None:
            best_pts = result.x.reshape((N, 2))
            best_score_val = -result.fun
            best_result = best_pts
    except Exception as e:
        # Fallback to regular polygon if DE fails
        pass
    
    # If DE didn't run or didn't find anything, use regular polygon
    if best_result is None:
        best_result = _initial_construction_regular()
        best_score_val = _score(best_result)
        
    # We can try a second phase: Local refinement using Nelder-Mead on the best DE result
    # This combines Global (DE) + Local (NM) search.
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        # Start from DE result
        x0 = best_result.flatten()
        
        # Nelder-Mead on the 28D space
        # This might be slow if not careful, but we have time.
        # Actually, NM in high dimensions can be tricky.
        # Let's stick to the DE result as it's likely near-optimal.
        # Or try a few local refinements from DE result.
        
        # Let's try one NM refinement.
        try:
            res_nm = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < -best_score_val: # i.e. -score < -best_score => score > best_score
                best_pts = res_nm.x.reshape((N, 2))
                best_score_val = -res_nm.fun
                best_result = best_pts
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END
