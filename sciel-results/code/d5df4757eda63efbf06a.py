"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Check for duplicate or nearly duplicate points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        # Cross product magnitude
        dx1 = pts[j, 0] - pts[i, 0]
        dy1 = pts[j, 1] - pts[i, 1]
        dx2 = pts[k, 0] - pts[i, 0]
        dy2 = pts[k, 1] - pts[i, 1]
        cross = abs(dx1 * dy2 - dy1 * dx2)
        area = 0.5 * cross
        if area < min_area:
            min_area = area
    
    return min_area / hull_area


def _score_params(params):
    """Score function for differential evolution.
    params: array of length 2*N with x,y coordinates
    """
    pts = params.reshape(N, 2)
    return -_score_points(pts)  # Negate for minimization


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Initial best: regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = init_pts.copy()
    best_score = _score_points(best_pts)
    
    # Differential evolution over raw coordinates
    # Bounds: points in [-1, 1] x [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    # Try differential evolution
    try:
        de_result = differential_evolution(
            _score_params,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            init='sobol',
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if de_result.fun < -best_score:
            de_pts = de_result.x.reshape(N, 2)
            de_score = _score_points(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Local refinement using Nelder-Mead
    try:
        nm_result = minimize(
            _score_params,
            best_pts.reshape(-1),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        nm_pts = nm_result.x.reshape(N, 2)
        nm_score = _score_points(nm_pts)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts.copy()
            
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 10:
        return best_pts
    
    # Try a few more random restarts with local refinement if time permits
    rng = np.random.RandomState(123)
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        
        # Random initialization
        random_pts = rng.uniform(-1, 1, size=(N, 2))
        
        try:
            nm_result = minimize(
                _score_params,
                random_pts.reshape(-1),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            nm_pts = nm_result.x.reshape(N, 2)
            nm_score = _score_points(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
                
        except Exception:
            continue
    
    return best_pts
# EVOLVE-BLOCK-END
