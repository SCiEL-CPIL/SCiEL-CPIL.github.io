"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices points[i], points[j], points[k]
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
        # Fallback: simple polygon area if all points are on hull
        return 1.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def construct():
    """
    Use Simulated Annealing to find 14 points in the unit square that
    maximize the minimum triangle area relative to the convex hull area.
    """
    # Time limit: 110 seconds to be safe
    time_limit = 110.0
    start_time = time.time()
    
    # Initial random configuration
    current_points = np.random.uniform(0, 1, (N, 2))
    current_score = _score(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # We will run SA for a large number of iterations, but check time
    max_iter = 100000
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Perturb current points
        # Randomly select a point to move
        idx = rng.integers(0, N)
        # Step size decreases slightly with temperature or just fixed
        step_size = 0.01 + 0.05 * T
        
        new_points = current_points.copy()
        # Move the selected point
        new_points[idx] += rng.normal(0, step_size, 2)
        
        # Keep within bounds [0, 1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        
        new_score = _score(new_points)
        
        # Acceptance probability
        if new_score > current_score:
            # Better solution, always accept
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_points = new_points.copy()
                best_score = current_score
        else:
            # Worse solution, accept with probability exp((new_score - current_score) / T)
            # Note: scores are positive. We want to maximize.
            # Delta = new_score - current_score (negative)
            delta = new_score - current_score
            if delta < 0:
                # Avoid division by zero or underflow
                if T > 0:
                    prob = np.exp(delta / T)
                else:
                    prob = 0.0
                
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final local refinement using Nelder-Mead on the best point found
    # This helps smooth out the SA result
    best_score = _score(best_points)
    
    # Run a few Nelder-Mead refinements
    for _ in range(3):
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if -res.fun > best_score:
                best_points = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
