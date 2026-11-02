"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

DEADLINE = 150.0  # seconds wall-clock limit
SAFETY_MARGIN = 5.0  # return this many seconds before deadline


def _min_triangle_ratio(points):
    """Compute min triangle area / convex hull area for given points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-15:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Triangle area = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    
    if min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _neg_objective(flat):
    """Negative of the score for minimization."""
    points = flat.reshape(-1, 2)
    return -_min_triangle_ratio(points)


def _initial_grid():
    """Generate a reasonable initial grid-like configuration."""
    # Start with points on a grid inside a unit square
    # Then we can optimize
    n = N
    # Use a roughly uniform distribution
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                points.append([x, y])
    
    return np.array(points[:n])


def construct():
    """Construct 14 points to maximize min triangle area / hull area."""
    start_time = time.time()
    
    # Initial configuration
    best_points = _initial_grid()
    best_score = _min_triangle_ratio(best_points)
    
    # Define bounds for differential evolution
    # Points in [0, 1] x [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Use differential evolution with multiple restarts
    # Each restart uses a different initial population strategy
    
    n_restarts = 3
    for restart in range(n_restarts):
        # Check deadline
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # For each restart, use a different strategy
        if restart == 0:
            # Standard DE
            init = 'latinhypercube'
            popsize = 15
        elif restart == 1:
            # Different strategy
            init = 'sobol'
            popsize = 20
        else:
            # Another strategy
            init = 'halton'
            popsize = 25
        
        try:
            result = differential_evolution(
                _neg_objective,
                bounds=bounds,
                seed=42 + restart,
                maxiter=200,
                tol=1e-6,
                popsize=popsize,
                init=init,
                polish=True,
                updating='immediate',
                seed=42 + restart
            )
            
            if result.success or result.fun < -best_score:
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _min_triangle_ratio(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
                    
        except Exception:
            pass
        
        # Check deadline
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    return best_points
# EVOLVE-BLOCK-END
