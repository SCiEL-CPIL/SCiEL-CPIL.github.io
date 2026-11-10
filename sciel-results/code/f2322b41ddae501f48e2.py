# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

_START_TIME = time.time()
_DEADLINE = _START_TIME + 570.0  # Safety margin before 630s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_triangle_area(points, beta=100.0):
    """
    Soft-min of triangle areas using log-sum-exp (smooth surrogate for min).
    As beta -> inf, this approaches the true min.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * areas)))
    # Use log-sum-exp trick for numerical stability
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + (-beta * max_area)
    soft_min = -(1.0 / beta) * log_sum
    
    return soft_min

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Score = A_min / A_hull
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min <= 1e-10:
        return 0.0
    
    return a_min / a_hull

def _objective(x):
    """
    Objective function to minimize (negative of the score).
    Score = A_min / A_hull
    """
    points = x.reshape(13, 2)
    return -_score(points)

def _objective_smooth(x, beta=100.0):
    """
    Smooth objective using soft-min for the triangle area.
    This is differentiable and suitable for gradient-based methods.
    """
    points = x.reshape(13, 2)
    a_min_soft = _soft_min_triangle_area(points, beta)
    a_hull = _hull_area(points)
    
    if a_hull <= 1e-10 or a_min_soft <= 1e-10:
        return 1.0  # Large positive value (we're minimizing)
    
    return -(a_min_soft / a_hull)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    
    Uses a multi-phase optimization strategy:
    1. Global search with Differential Evolution on the exact objective.
    2. Local refinement with Powell's method on the exact objective.
    3. Gradient polish with L-BFGS-B on a soft-min surrogate.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Define bounds for the search space
    # Normalizing to [-1, 1] is a good starting point, but the optimal solution
    # might be better scaled. We can let DE search in a slightly larger box.
    bounds = [(-1.5, 1.5)] * (2 * n)
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust to non-smooth functions and can escape local minima.
    # We use a moderate popsize and maxiter to balance exploration and time.
    if time.time() < _DEADLINE - 120:
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                polish=True,  # Use L-BFGS-B to polish the DE result
                updating='immediate',
                workers=-1  # Use all available cores
            )
            
            if de_result.success or de_result.fun < best_score:
                candidate_points = de_result.x.reshape(n, 2)
                candidate_score = _score(candidate_points)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
                    
        except Exception:
            pass
    
    # If DE didn't find anything or we are still early, try structured starts with Powell
    if best_points is None or time.time() < _DEADLINE - 60:
        # Define several structured ring topologies for 13 points
        ring_configs = [
            [(0, 1), (0.5, 6), (1.0, 6)],           # center + 2 rings
            [(0, 1), (0.4, 4), (0.8, 4), (1.2, 4)], # center + 3 rings
            [(0, 1), (0.3, 3), (0.6, 3), (0.9, 3), (1.2, 3)],  # center + 4 rings
            [(0.3, 5), (0.8, 8)],                     # 2 rings only
            [(0.2, 3), (0.5, 5), (1.0, 5)],           # 3 rings
            [(0, 1), (0.7, 12)],                      # center + 1 ring
            [(0.4, 6), (0.9, 7)],                     # 2 rings
            [(0, 1), (0.3, 5), (0.7, 7)],             # center + 2 rings
        ]
        
        for i in range(min(20, len(ring_configs) * 3)):
            if time.time() > _DEADLINE - 60:
                break
            
            config_idx = i % len(ring_configs)
            ring_config = ring_configs[config_idx]
            
            # Add random angle offsets
            angle_offsets = [rng.random() * 2 * np.pi for _ in ring_config]
            
            points = []
            for ring_idx, (radius, num_points) in enumerate(ring_config):
                offset = angle_offsets[ring_idx]
                for k in range(num_points):
                    angle = 2 * np.pi * k / num_points + offset
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    points.append([x, y])
            
            x0 = np.array(points).flatten()
            
            # Powell optimization on exact objective
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 2000,
                        'xtol': 1e-10,
                        'ftol': 1e-12,
                    }
                )
                
                current_pts = res.x.reshape(n, 2)
                current_score = _score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
            except Exception:
                continue
    
    # Phase 2: Local Refinement with Powell
    if best_points is not None and time.time() < _DEADLINE - 30:
        try:
            res = minimize(
                _objective,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xtol': 1e-12,
                    'ftol': 1e-14,
                }
            )
            
            current_pts = res.x.reshape(n, 2)
            current_score = _score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception:
            pass
    
    # Phase 3: Gradient Polish with L-BFGS-B on soft-min surrogate
    # Use a high beta to approximate the min closely while maintaining differentiability
    if best_points is not None and time.time() < _DEADLINE - 10:
        try:
            # Try a few betas for progressive refinement
            for beta in [100.0, 500.0, 1000.0]:
                if time.time() > _DEADLINE - 5:
                    break
                
                res = minimize(
                    lambda x: _objective_smooth(x, beta=beta),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                    }
                )
                
                current_pts = res.x.reshape(n, 2)
                current_score = _score(current_pts)
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
                    
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: random points
        best_points = rng.random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
