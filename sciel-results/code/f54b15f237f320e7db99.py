import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    n = len(points)
    indices = list(combinations(range(n), 3))
    if not indices:
        return np.array([])
    
    # Extract coordinates for all triangles
    # This is memory efficient enough for N=13
    i_idx = np.array([p[0] for p in indices])
    j_idx = np.array([p[1] for p in indices])
    k_idx = np.array([p[2] for p in indices])
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Area = 0.5 * |(xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)|
    # Vectorized cross product in 2D
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(params):
    """Objective function for minimization (direct score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_points = None
    
    # Phase 1: Differential Evolution for global search
    # Bounds: points in [-1, 1]^2
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        de_result = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=300,
            popsize=15,
            tol=1e-8,
            mutation=(0.3, 1.2),
            recombination=0.8,
            init='latinhypercube',
            polish=False,  # We'll do our own polishing
            update_interval=1,
            callback=None
        )
        de_pts = de_result.x.reshape(N, 2)
        de_score = _score(de_pts)
        if de_score > best_score:
            best_score = de_score
            best_points = de_pts
    except Exception:
        pass
    
    # Phase 2: Local refinement with Nelder-Mead on best DE result
    if best_points is not None:
        if time.time() < deadline - 5:
            try:
                nm_result = minimize(
                    _neg_score,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                nm_pts = nm_result.x.reshape(N, 2)
                nm_score = _score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_pts
            except Exception:
                pass
    
    # Phase 3: Try regular polygon and nearby perturbations
    if time.time() < deadline - 3:
        angles = 2 * np.pi * np.arange(N) / N
        regular = np.column_stack([np.cos(angles), np.sin(angles)])
        reg_score = _score(regular)
        if reg_score > best_score:
            best_score = reg_score
            best_points = regular
        
        # Perturb regular polygon
        rng = np.random.RandomState(123)
        for trial in range(10):
            if time.time() > deadline - 2:
                break
            noise = rng.normal(0, 0.05, (N, 2))
            perturbed = regular + noise
            try:
                nm_result = minimize(
                    _neg_score,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                nm_pts = nm_result.x.reshape(N, 2)
                nm_score = _score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_pts
            except Exception:
                continue
    
    # Phase 4: If still not good, try more diverse starts
    if best_score < 0.05 and time.time() < deadline - 5:
        for seed in range(20):
            if time.time() > deadline - 3:
                break
            rng = np.random.RandomState(seed * 7 + 3)
            # Mix of circular and random
            if seed % 2 == 0:
                angles = 2 * np.pi * np.arange(N) / N + rng.uniform(-0.15, 0.15, N)
                radii = 1.0 + rng.uniform(-0.3, 0.3, N)
                init_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            else:
                init_pts = rng.uniform(-1, 1, (N, 2))
            
            try:
                nm_result = minimize(
                    _neg_score,
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                nm_pts = nm_result.x.reshape(N, 2)
                nm_score = _score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_pts
            except Exception:
                continue
    
    if best_points is None:
        # Fallback to regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
