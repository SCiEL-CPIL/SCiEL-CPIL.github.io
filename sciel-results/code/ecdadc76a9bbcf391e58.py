import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _objective(params):
    """
    Objective function for minimizers.
    Returns negative of the score (min triangle area / hull area).
    Normalizes points to unit hull area to remove scale/translation invariance.
    """
    try:
        pts = params.reshape(N, 2)
        
        # Center
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        
        # Check for degenerate hull
        try:
            hull = ConvexHull(pts_centered)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e10
        except Exception:
            return 1e10
        
        # Scale to unit area
        scale = np.sqrt(1.0 / hull_area)
        pts_norm = pts_centered * scale
        
        # Calculate score
        score = _exact_score(pts_norm)
        
        # Return negative for minimization
        return -score
        
    except Exception:
        return 1e10

def construct():
    t0 = time.time()
    deadline = 145.0  # Safety margin
    
    # Bounds for Differential Evolution
    # Points can be anywhere, but normalization handles scale.
    # A box of [-10, 10] is sufficient for the relative geometry.
    bounds = [(-10.0, 10.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust to non-smooth objectives and finds global basins well.
    # We allocate ~60% of the time to DE.
    de_time_budget = 90.0
    
    if time.time() - t0 < deadline - de_time_budget:
        try:
            # popsize and maxiter control the number of function evaluations.
            # Each evaluation involves hull computation and triangle area computation.
            # C(13,3) = 286 triangles. This is fast.
            # Let's try popsize=15, maxiter=50. 15*50 = 750 evaluations.
            # This should be very fast, leaving time for polishing.
            res_de = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False # We will polish separately
            )
            
            if res_de.success or res_de.fun < 1e9:
                best_points = res_de.x.reshape(N, 2)
                best_score = _exact_score(best_points)
                
        except Exception:
            pass
    
    # If DE failed or didn't run, use a random start
    if best_points is None:
        rng = np.random.RandomState(42)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.2, 0.2, N)
        radii = 1.0 + rng.uniform(-0.2, 0.2, N)
        best_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        best_score = _exact_score(best_points)

    # Phase 2: Local Polish with Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smooth functions well.
    # It is much faster per iteration than DE for local refinement.
    remaining_time = deadline - (time.time() - t0)
    
    if remaining_time > 5.0:
        try:
            # Run Nelder-Mead for the remaining time
            # maxiter is high, but it will stop if time runs out or convergence is reached.
            # We don't have a direct time limit in scipy.optimize.minimize, 
            # so we rely on maxiter and ftol.
            # Let's set a generous maxiter and tight tolerances.
            res_nm = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < 1e9:
                pts_nm = res_nm.x.reshape(N, 2)
                score_nm = _exact_score(pts_nm)
                
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm
        
        except Exception:
            pass
            
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
