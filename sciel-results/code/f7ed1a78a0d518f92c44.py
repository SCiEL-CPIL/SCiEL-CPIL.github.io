"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 110.0  # safety margin below 120s limit


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _hull_area(points)
    if hull_a <= 1e-12:
        return 0.0
    return min_ta / hull_a


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _objective(params):
    """Objective function for optimizer: negative of score."""
    points = params.reshape(N, 2)
    return _negative_score(points)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def construct():
    start_time = time.time()
    
    # Start with regular 13-gon
    points = _regular_polygon(N, radius=1.0)
    best_score = _score(points)
    best_points = points.copy()
    
    # Local optimization using L-BFGS-B
    x0 = points.flatten()
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000}
        )
        
        if result.success or result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            candidate_score = _score(candidate)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate.copy()
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_points
    
    # Try a few more optimization passes with different initializations
    for restart in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Slightly perturb the best point
        if restart == 0:
            init = best_points.copy()
        else:
            init = _regular_polygon(N, radius=1.0 + 0.1 * restart)
            # Add small random perturbation
            rng = np.random.RandomState(restart * 1000 + 42)
            init += rng.normal(0, 0.01, init.shape)
        
        x0 = init.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 5000}
            )
            
            if result.success or result.fun < -best_score:
                candidate = result.x.reshape(N, 2)
                candidate_score = _score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points
