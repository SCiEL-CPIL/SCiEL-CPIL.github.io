"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 15.0  # seconds before deadline to stop

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _renormalize_unit_hull(pts):
    """
    Rescales points so that their convex hull has area exactly 1.0.
    Also translates the centroid to (0,0) to reduce parameter space drift.
    Returns None if hull area is invalid.
    """
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return None
    
    # Scale factor: Area scales with scale^2.
    # We want New_Area = 1.0
    # New_Area = scale^2 * Old_Area
    # scale = 1 / sqrt(Old_Area)
    scale = 1.0 / np.sqrt(hull_area)
    
    # Translate to center
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    pts_scaled = pts_centered * scale
    
    return pts_scaled

def _objective(x):
    """
    Objective function for minimization (we minimize negative score).
    x: flattened array of 2N coordinates.
    """
    # Check time limit
    if time.time() - _START > _DEADLINE - _SAFETY:
        return 0.0
        
    pts = x.reshape(N, 2)
    
    # Renormalize to fix scale
    pts_norm = _renormalize_unit_hull(pts)
    if pts_norm is None:
        return 0.0
        
    # Calculate score on normalized points
    # Since hull area is 1, score is just min_triangle_area
    min_area = _min_triangle_area_exact(pts_norm)
    
    # Return negative score for minimization
    return -min_area

def _best_known_config():
    """
    Returns a high-quality initial guess.
    Uses a regular 13-gon perturbed slightly, or a known good structure.
    """
    # Start with a regular 13-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Add small random perturbation to break symmetry and help DE escape
    rng = np.random.RandomState(42)
    pts += rng.randn(N, 2) * 0.01
    
    # Normalize
    return _renormalize_unit_hull(pts)

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Not enough time for DE, return best static guess
        return _best_known_config()

    # Define bounds for the initial DE population.
    # Since we renormalize, the absolute scale doesn't matter, but we need
    # a reasonable range to start. [-2, 2] is safe for unit-area hulls.
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Initial population
    init_pts = _best_known_config()
    if init_pts is None:
        init_pts = np.random.rand(N, 2) * 4 - 2
    x0 = init_pts.flatten()
    
    # Differential Evolution Parameters
    # maxiter needs to be high enough to converge, but we have a time limit.
    # popsize: 15 (standard)
    # tol: 1e-8
    # mutation: (0.5, 1.0) standard
    # recombination: 0.9
    
    # We will run DE. The callback allows us to check time.
    # However, DE's callback in SciPy doesn't easily allow stopping mid-iteration 
    # without complex threading or just checking time in the objective.
    # The objective function checks time and returns 0.0 (bad score) if time is up,
    # which effectively stalls progress, but DE might still waste time on evaluations.
    
    # To be safer, we can limit maxiter based on remaining time.
    # Assume ~0.1s per evaluation (optimistic for 26D).
    # DE evaluations per iteration = popsize * 2 (roughly).
    # Let's just set a reasonable maxiter and rely on the objective time-check.
    # If time runs out, DE will finish with the best found so far if we handle it,
    # but standard DE returns the best at the end of maxiter or convergence.
    
    # Strategy: Run DE with a maxiter that is likely to exceed time limit,
    # but rely on the fact that DE is stochastic and might find a good solution early.
    # Actually, a better strategy for time-limited DE is to run it in chunks or
    # just let it run and catch the timeout? No, we must return within 120s.
    
    # Let's estimate:
    # 13 points -> 26 dims.
    # 286 triangles.
    # Vectorized min area is fast.
    # ConvexHull is the bottleneck.
    # Let's aim for ~1000-2000 evaluations.
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            x0=x0,
            seed=42,
            maxiter=200, 
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False, # Polish uses L-BFGS which we are avoiding as primary strategy, 
                          # but a final polish might be good if time permits.
                          # However, polish can be slow. Let's do manual polish later.
            disp=False
        )
        
        best_x = result.x
        best_pts = best_x.reshape(N, 2)
        best_pts = _renormalize_unit_hull(best_pts)
        
    except Exception as e:
        # Fallback to initial guess if DE fails
        best_pts = _best_known_config()
        if best_pts is None:
             # Absolute fallback
             angles = 2 * np.pi * np.arange(N) / N
             best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
             best_pts = _renormalize_unit_hull(best_pts)

    # If we have time left, perform a local polish using Nelder-Mead
    # on the normalized points. Nelder-Mead is derivative-free and robust.
    if time.time() - _START < _DEADLINE - _SAFETY - 5.0:
        from scipy.optimize import minimize
        
        # Objective for local polish: minimize negative min area (hull is fixed to 1 by renorm inside)
        def polish_obj(x):
            if time.time() - _START > _DEADLINE - _SAFETY:
                return 0.0
            pts = x.reshape(N, 2)
            pts_norm = _renormalize_unit_hull(pts)
            if pts_norm is None:
                return 0.0
            return -_min_triangle_area_exact(pts_norm)
            
        try:
            # Nelder-Mead is good for low-to-medium dimensional non-smooth problems
            res = minimize(polish_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.success:
                polished_pts = res.x.reshape(N, 2)
                polished_pts = _renormalize_unit_hull(polished_pts)
                if polished_pts is not None:
                    # Verify it's better
                    s_orig = _score(best_pts)
                    s_pol = _score(polished_pts)
                    if s_pol > s_orig:
                        best_pts = polished_pts
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
