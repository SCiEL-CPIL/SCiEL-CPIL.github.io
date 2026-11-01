"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate minimum triangle area
    min_area = float('inf')
    # Iterate over all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross(p2-p1, p3-p1)|
                cross = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def construct():
    # Simulated Annealing Approach
    # 1. Initialization: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.005  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    steps = 50000
    sigma = 0.1 # Step size
    
    for i in range(steps):
        # Cool
        T *= cooling_rate
        if T < T_min:
            break
            
        # Pick random point
        idx = np.random.randint(N)
        old_point = points[idx].copy()
        
        # Perturb
        points[idx] = old_point + np.random.normal(0, sigma, 2)
        
        new_score = _score(points)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            delta = new_score - current_score
            # Avoid log(0) or overflow
            if delta < -100 * T: # If delta is very negative, exp is effectively 0
                # Revert
                points[idx] = old_point
            else:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    points[idx] = old_point
                    
    return best_points
# EVOLVE-BLOCK-END
