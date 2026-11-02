"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # For 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Compute the ratio: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective(x):
    """Objective for differential_evolution: we minimize, so return negative score."""
    points = x.reshape(-1, 2)
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Bounds: points in [-2, 2] x [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    best_x = x0.copy()
    best_score = _score(x0.reshape(-1, 2))
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=x0,
            seed=42,
            maxiter=500,
            tol=1e-8,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            init='latinhypercube',
            callback=lambda xk, convergence: (
                None if time.time() < deadline else (_stop_de())
            ),
        )
        
        if result.success or result.fun < _objective(best_x):
            best_x = result.x
            best_score = _score(best_x.reshape(-1, 2))
    except Exception:
        pass
    
    # If we have time, try a few local refinements
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            for _ in range(5):
                if time.time() >= deadline - 5:
                    break
                res = minimize(
                    _objective,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < _objective(best_x):
                    best_x = res.x
                    best_score = _score(best_x.reshape(-1, 2))
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)


def _stop_de():
    """Helper to signal stopping differential_evolution."""
    raise ValueError("Deadline reached")

# EVOLVE-BLOCK-END
