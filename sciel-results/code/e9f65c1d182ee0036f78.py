import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns a scalar. Lower is better for minimization (we will negate it).
    """
    try:
        # Calculate convex hull area
        # Ensure points are not degenerate
        if len(points) < 3:
            return 1e6
        
        # Check if all points are collinear or identical
        # ConvexHull requires at least 3 non-collinear points
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume # In 2D, volume is area
        except Exception:
            # If hull fails, points are likely collinear or degenerate
            # Return a high value (bad score) to avoid this configuration
            return 1e6
            
        if hull_area <= 1e-9:
            return 1e6

        # Calculate all triangle areas
        # Number of points N
        n = len(points)
        # Indices for all triples i < j < k
        i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1).T
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        # Area = 0.5 * |(pj - pi) x (pk - pi)|
        # In 2D, cross product is scalar: (x1*y2 - y1*x2)
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / hull_area

    except Exception:
        return 1e6

def _neg_score(points_flat):
    """
    Objective function for minimization.
    Takes a flat array of 2*N floats, reshapes to (N, 2), and returns -score.
    """
    points = points_flat.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = -1e6
    best_points = None
    
    # Initial guess: Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    regular_poly = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # List of initial points to try
    initial_candidates = [regular_poly]
    
    # Add perturbations of the regular polygon
    rng = np.random.default_rng(42)
    for _ in range(50):
        perturbation = rng.normal(0, 0.1, size=(N, 2))
        candidate = regular_poly + perturbation
        initial_candidates.append(candidate)
        
    # Add some random points in the unit square (normalized roughly)
    for _ in range(20):
        candidate = rng.uniform(0, 1, size=(N, 2))
        # Center and scale to unit area roughly? 
        # The score is invariant to affine transformations? 
        # No, the score is min_tri_area / hull_area.
        # If we scale all points by k, areas scale by k^2, so ratio is invariant.
        # If we translate, areas unchanged, hull area unchanged.
        # So we can just use random points in [0,1]^2.
        initial_candidates.append(candidate)

    # Shuffle candidates to ensure diversity in time
    rng.shuffle(initial_candidates)
    
    for candidate in initial_candidates:
        if time.time() > deadline:
            break
            
        # Try to optimize from this candidate
        # Nelder-Mead is good for small dimensions and non-smooth objectives
        try:
            res = minimize(
                _neg_score, 
                candidate.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.success or res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass

    # If no optimization succeeded, return the best initial candidate
    if best_points is None:
        # Fallback to regular polygon
        best_points = regular_poly

    return best_points
