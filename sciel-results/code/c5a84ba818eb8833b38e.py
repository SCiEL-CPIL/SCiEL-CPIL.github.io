"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import math

# Precompute all triple indices for N points
# This allows vectorized area calculation
_TRIPLES = np.array(list(itertools.combinations(range(N), 3)))

def _score_vectorized(points):
    """
    Calculate min triangle area / convex hull area using vectorization.
    points: (N, 2) array
    """
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
        
    # 2. Calculate all triangle areas vectorized
    # _TRIPLES shape is (364, 3)
    p0 = points[_TRIPLES[:, 0]]
    p1 = points[_TRIPLES[:, 1]]
    p2 = points[_TRIPLES[:, 2]]
    
    # Cross product in 2D
    dx1 = p1[:, 0] - p0[:, 0]
    dy1 = p1[:, 1] - p0[:, 1]
    dx2 = p2[:, 0] - p0[:, 0]
    dy2 = p2[:, 1] - p0[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    # Simulated Annealing
    # Initial points in unit square [-1, 1]
    points = np.random.uniform(-1, 1, (N, 2))
    current_score = _score_vectorized(points)
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.0001
    iterations = 20000
    # Cooling factor
    alpha = (T_end / T_start) ** (1.0 / iterations)
    T = T_start
    step_size = 0.2
    
    for i in range(iterations):
        # Generate move: perturb 1 random point
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        
        # Gaussian perturbation
        noise = np.random.normal(0, step_size, 2)
        points[idx] += noise
        
        # Keep within bounds
        points[idx] = np.clip(points[idx], -2, 2)
        
        new_score = _score_vectorized(points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                if T > 1e-10:
                    prob = math.exp(delta / T)
                else:
                    prob = 0.0
            else:
                prob = 1.0
            
            if np.random.random() < prob:
                current_score = new_score
            else:
                # Revert
                points[idx] = old_point
        
        # Cool down
        T *= alpha
        # Decay step size to fine-tune
        step_size *= 0.9998
        
    return best_points
# EVOLVE-BLOCK-END
