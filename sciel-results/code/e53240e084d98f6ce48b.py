"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Pre-compute all triangle index combinations for vectorization
_TRI_INDICES = np.array(list(combinations(range(N), 3)))

def _min_triangle_area_vec(points):
    """
    Compute the minimum area over all triples of points.
    Vectorized for speed.
    """
    if len(points) < 3:
        return 0.0
    # Extract coordinates for each triangle vertex
    pi = points[_TRI_INDICES[:, 0]]
    pj = points[_TRI_INDICES[:, 1]]
    pk = points[_TRI_INDICES[:, 2]]
    
    # Compute cross product for area
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area_fast(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """
    Score = min triangle area / convex hull area.
    Higher is better.
    """
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(points)
    return min_tri / hull_area

def _points_from_ellipse(params):
    """
    Convert elliptical parameters (14 angles, a, b) to Cartesian points.
    params: array of shape (16,)
    """
    angles = params[:N]
    a = params[N]
    b = params[N+1]
    
    # Ensure positive semi-axes
    a = max(abs(a), 0.01)
    b = max(abs(b), 0.01)
    
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    return np.column_stack((x, y))

def _objective_ellipse(params):
    """
    Objective function for elliptical parameterization.
    Minimizes negative score.
    """
    points = _points_from_ellipse(params)
    score = _score(points)
    return -score

def _objective_free(flat_points):
    """
    Objective function for free 2D points.
    Minimizes negative score.
    """
    points = flat_points.reshape(-1, 2)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial guess: Regular 14-gon on unit circle (a=1, b=1)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rng = np.random.default_rng(42)
    
    best_score_val = 0.0
    best_points = None
    
    # --- Phase 1: Optimize Elliptical Parameters ---
    # Try multiple random perturbations of the regular polygon angles
    # to escape local minima.
    
    max_ellipse_trials = 10
    for trial in range(max_ellipse_trials):
        if time.time() > deadline - 30.0:  # Leave time for refinement
            break
        
        if trial == 0:
            angles0 = base_angles.copy()
        else:
            # Perturb angles slightly. Larger perturbations for later trials?
            # Keep it small to maintain convexity/order roughly
            noise_std = 0.05 * (1 + trial * 0.2)
            angles0 = base_angles + rng.normal(0, noise_std, N)
            # Sort angles to ensure they are in order around the ellipse
            angles0 = np.sort(angles0) % (2 * np.pi)
        
        # Initial a, b. Start with circle.
        params0 = np.concatenate([angles0, [1.0, 1.0]])
        
        try:
            res = minimize(
                _objective_ellipse,
                params0,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.fun < 0:
                current_score = -res.fun
                if current_score > best_score_val:
                    best_score_val = current_score
                    best_points = _points_from_ellipse(res.x)
        except Exception:
            pass

    # Fallback if no elliptical optimization succeeded
    if best_points is None:
        best_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        best_score_val = _score(best_points)
    
    # --- Phase 2: Free 2D Refinement ---
    # Take the best elliptical result and allow points to move freely in 2D.
    # This can improve the score by breaking the strict elliptical constraint
    # if the hull area normalization allows for a better packing.
    
    if time.time() < deadline - 5.0:
        x0_free = best_points.flatten()
        
        try:
            res_free = minimize(
                _objective_free,
                x0_free,
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_free.fun < 0:
                current_score = -res_free.fun
                if current_score > best_score_val:
                    best_score_val = current_score
                    best_points = res_free.x.reshape(-1, 2)
        except Exception:
            pass

    # Final sanity check: ensure points are valid
    if best_points is None or len(best_points) != N:
        best_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    return best_points

# EVOLVE-BLOCK-END
