"""
Strategy: General 28-dimensional optimization.
Instead of restricting to symmetric rings (7+7), we optimize the coordinates
of all 14 points directly. We use a fast vectorized scoring function and
combine random restarts with local Nelder-Mead optimization to escape local
minima and refine the best candidate.
"""

import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Vectorized for speed.
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
    # Vectorized calculation for all triples
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

def _negative_score(params):
    """
    Objective function for optimization.
    params: flat array of shape (28,) representing 14 (x, y) points.
    """
    pts = params.reshape(N, 2)
    # Add a small penalty if points are too close or hull is degenerate
    # to help the optimizer avoid singularities, though _score handles it.
    # A simple check for uniqueness might be useful but _score returns 0 for degenerate.
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin before 150s limit
    
    best_score = -float('inf')
    best_params = None
    
    # Strategy:
    # 1. Generate several random initial configurations (normalized to unit area roughly).
    #    We can use random points in a circle or square, or perturb a symmetric base.
    #    Since we want to break symmetry, random points in a circle are a good start.
    # 2. Run local Nelder-Mead optimization from these starts.
    # 3. Keep the best result.
    
    rng = np.random.default_rng(42)
    
    # Initial guesses:
    # 1. Regular 14-gon (high symmetry, might be a local optimum but a good reference)
    # 2. Two rings (7+7) - the previous strategy's best guess
    # 3. Random points in a circle
    # 4. Random points in a square
    
    initial_guesses = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_pts.flatten())
    
    # 2. Two rings (7 outer, 7 inner)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    ring_pts = np.vstack([outer_pts, inner_pts])
    initial_guesses.append(ring_pts.flatten())
    
    # 3. Random points in unit circle
    for _ in range(10):
        # Rejection sampling for circle
        while True:
            r = rng.random((N, 2)) * 2 - 1
            if np.all(np.sum(r**2, axis=1) < 1):
                break
        initial_guesses.append(r.flatten())
        
    # 4. Random points in unit square
    for _ in range(10):
        sq_pts = rng.random((N, 2))
        initial_guesses.append(sq_pts.flatten())
        
    # Sort initial guesses by score to prioritize better starts?
    # Or just iterate through them. Given time constraints, we can't evaluate all if Nelder-Mead is slow.
    # Let's evaluate the score of each guess first to prioritize.
    
    scored_guesses = []
    for guess in initial_guesses:
        pts = guess.reshape(N, 2)
        s = _score(pts)
        scored_guesses.append((s, guess))
        
    # Sort by score descending
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # We will try to optimize the top K guesses.
    # How many can we do? Each Nelder-Mead run might take ~1-5 seconds depending on convergence.
    # With 145s budget, we can afford ~20-30 runs if they are fast, or fewer if slow.
    # Let's try the top 20.
    
    max_attempts = 20
    
    for i, (s_guess, guess) in enumerate(scored_guesses[:max_attempts]):
        if time.time() > deadline - 10.0: # Leave time for final check
            break
            
        if s_guess > best_score:
            best_score = s_guess
            best_params = guess.copy()
            
        # Run local optimization
        try:
            res = minimize(_negative_score, guess, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 1000, 'maxfev': 1000})
            
            if res.fun < -best_score: # res.fun is negative score, so lower is better
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_params = res.x.copy()
        except Exception:
            pass
            
    # If we still have time, we can try a few more random restarts
    # Or refine the best found further?
    if best_params is not None and time.time() < deadline - 5.0:
         # Try to refine the best found with a tighter tolerance
         try:
            res = minimize(_negative_score, best_params, method='Nelder-Mead', 
                           options={'xatol': 1e-7, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x.copy()
         except Exception:
            pass

    # Fallback if something went wrong
    if best_params is None:
        best_params = initial_guesses[0]
        
    final_pts = best_params.reshape(N, 2)
    
    # Optional: Normalize to unit area? 
    # The task says "normalized to unit area" but the score is min_area / hull_area.
    # Scaling the points scales both numerator and denominator by the same factor (scale^2).
    # So the ratio is invariant to scaling.
    # However, the evaluator might expect points in a specific range or just the ratio.
    # The contract says: "score is the minimum triangle area divided by the convex hull area".
    # So scaling doesn't matter for the score.
    # But returning points with very large or small coordinates might be bad practice.
    # Let's normalize the hull area to 1 for the output, just to be clean.
    try:
        hull = ConvexHull(final_pts)
        area = hull.volume
        if area > 1e-9:
            scale = 1.0 / np.sqrt(area)
            final_pts = final_pts * scale
    except Exception:
        pass
        
    return final_pts
