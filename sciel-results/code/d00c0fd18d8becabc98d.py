"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (C(n,3),)."""
    n = len(points)
    # Precompute indices for all triples
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Vectorized cross product magnitude
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        # ConvexHull requires at least 3 non-collinear points in 2D
        if len(points) < 3:
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _objective(x):
    """Objective function for optimization. Input is flat array of 28 coordinates."""
    pts = x.reshape(-1, 2)
    # Add a penalty if hull area is degenerate to help DE
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-6:
        return 1e6
    return _negative_score(pts)

def _optimize_global(deadline, popsize=20, maxiter=100):
    """
    Run Differential Evolution over the full 28D space.
    Normalization invariance means absolute scale doesn't matter, but we bound
    coordinates to prevent numerical instability.
    """
    # Bounds: coordinates in [-1, 1] is sufficient due to normalization
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=42,
            polish=False, # We will do our own polish
            init='sobol'
        )
        if result.success or result.fun < 1e5:
            return result.x.reshape(-1, 2), _score(result.x.reshape(-1, 2))
    except Exception as e:
        pass
    return None, -1

def _optimize_local(points, deadline, method='Nelder-Mead', maxiter=5000):
    """Local optimization using Nelder-Mead."""
    def obj(x):
        return _negative_score(x.reshape(-1, 2))
    
    x0 = points.flatten()
    try:
        result = minimize(
            obj,
            x0,
            method=method,
            options={
                'maxiter': maxiter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        best_points = result.x.reshape(-1, 2)
        best_score = _score(best_points)
        return best_points, best_score
    except Exception:
        return points, _score(points)

def _fine_tune(points, deadline, max_iters=20000, initial_step=0.01):
    """
    Custom fine-tuning: random perturbations with adaptive step size.
    Helps escape very shallow local minima that Nelder-Mead might get stuck in.
    """
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    rng = np.random.RandomState(123)
    step_size = initial_step
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Perturb a random subset of points
        n_perturb = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        deltas = rng.normal(0, step_size, size=(n_perturb, 2))
        
        new_points = points.copy()
        new_points[idxs] += deltas
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            no_improve += 1
            if no_improve > 200:
                step_size *= 0.9
                no_improve = 0
                if step_size < 1e-6:
                    break
                    
    return best_points, best_score

def construct():
    start_time = time.time()
    # Safety margin: kill at 145s to be safe against 150s limit
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust to initial conditions and can handle the non-convex landscape
    if time.time() < deadline - 60:
        # Start with a moderate population and iterations
        # popsize=20, maxiter=100 is a good balance for 28D
        pts, sc = _optimize_global(deadline, popsize=20, maxiter=100)
        if pts is not None:
            best_points = pts
            best_score = sc
    
    # Fallback if DE failed or produced poor results
    if best_points is None or best_score < 0.001:
        # Initialize with a regular polygon as a decent starting point
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)

    # Phase 2: Local Refinement
    # Use Nelder-Mead to polish the global solution
    if time.time() < deadline - 30:
        pts, sc = _optimize_local(best_points, deadline, maxiter=5000)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Phase 3: Fine Tuning
    # Use custom random search to squeeze out more precision
    if time.time() < deadline - 10:
        pts, sc = _fine_tune(best_points, deadline, max_iters=15000, initial_step=0.05)
        if sc > best_score:
            best_points = pts
            best_score = sc
            
    # Phase 4: Final Polish
    # One last Nelder-Mead pass with tighter tolerances
    if time.time() < deadline - 2:
        pts, sc = _optimize_local(best_points, deadline, maxiter=2000)
        if sc > best_score:
            best_points = pts
            best_score = sc

    # Final check: ensure we return a valid array
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
