"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(pts):
    """Compute minimum triangle area for n points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _objective(x):
    """Objective to minimize: negative of score."""
    pts = x.reshape((-1, 2))
    return -_score(pts)

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial guess: Regular 13-gon
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Try a few other symmetric starts
    starts = [
        _regular_polygon(N, radius=0.45, center=(0.5, 0.5)),
        _regular_polygon(N, radius=0.55, center=(0.5, 0.5)),
        np.array([[0.5, 0.5]] + list(_regular_polygon(12, radius=0.5, center=(0.5, 0.5)))) # Center + 12-gon
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        s = _score(start)
        if s > best_score:
            best_score = s
            best_pts = start.copy()

    # Global Optimization using Differential Evolution
    # Bounds: [0, 1] for each coordinate
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # We need to be careful with time. DE is stochastic.
    # We'll run it for a limited number of iterations or time.
    # Since we can't easily pass a time limit to DE, we estimate.
    # One iteration of DE with popsize P and dim D takes roughly P * D * cost_eval.
    # cost_eval is small (N=13).
    # Let's try a moderate population size and maxiter.
    
    try:
        # Use a callback to stop if time is running out? 
        # scipy.optimize.differential_evolution doesn't support time-based stopping directly.
        # We will run it in a way that is likely to finish.
        # Popsize 15, maxiter 200. 15 * 200 = 3000 evaluations.
        # Each eval is ~microseconds. This should be fast.
        
        # However, DE can be slow if it gets stuck.
        # Let's run a shorter DE first, then NM.
        
        # Strategy:
        # 1. Run DE with maxiter=100, popsize=20
        # 2. Take result, run NM polish.
        # 3. If time remains, run another DE with different seed/params.
        
        result = differential_evolution(_objective, bounds, 
                                        maxiter=100, 
                                        popsize=20,
                                        seed=42,
                                        tol=1e-8,
                                        polish=True) # polish=True uses L-BFGS-B by default, which might be slow or fail. Let's set polish=False and do our own NM.
        
        if result.fun < -best_score:
            best_score = -result.fun
            best_pts = result.x.reshape((-1, 2))
            
    except Exception as e:
        pass

    # Local Refinement with Nelder-Mead
    # This is crucial for fine-tuning.
    if time.time() < deadline - 5.0:
        x0 = best_pts.flatten()
        try:
            result = minimize(_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                      'adaptive': True})
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape((-1, 2))
        except Exception:
            pass
            
    # If time remains, try another DE run with different seed
    if time.time() < deadline - 20.0:
        try:
            result = differential_evolution(_objective, bounds, 
                                            maxiter=100, 
                                            popsize=20,
                                            seed=123,
                                            tol=1e-8,
                                            polish=False)
            
            if result.fun < -best_score:
                best_score = -result.fun
                best_pts = result.x.reshape((-1, 2))
                
                # Polish again
                if time.time() < deadline - 5.0:
                    result2 = minimize(_objective, best_pts.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                              'adaptive': True})
                    if result2.fun < -best_score:
                        best_score = -result2.fun
                        best_pts = result2.x.reshape((-1, 2))
                        
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
