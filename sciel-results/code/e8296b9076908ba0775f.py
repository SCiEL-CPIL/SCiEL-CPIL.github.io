import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 14

def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area == 0:
        return 0.0

    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def greedy_grid_placement(grid_size, n_points):
    """
    Greedily place points on a grid to maximize the minimum triangle area.
    """
    # Generate grid points in [0, 1] x [0, 1]
    xs = np.linspace(0.1, 0.9, grid_size) # Avoid exact edges to keep hull well-defined
    ys = np.linspace(0.1, 0.9, grid_size)
    grid_points = np.array([[x, y] for x in xs for y in ys])
    
    placed = []
    remaining = grid_points.copy()
    
    # First point: center
    center_idx = np.argmin(np.sum((grid_points - 0.5)**2, axis=1))
    placed.append(grid_points[center_idx])
    remaining = np.delete(remaining, center_idx, axis=0)
    
    for _ in range(1, n_points):
        best_score = -1
        best_idx = -1
        # Evaluate each candidate
        for i, p in enumerate(remaining):
            # Calculate min triangle area with all pairs of placed points
            # If < 2 placed points, we can't form a triangle yet, so score is high/infinite
            if len(placed) < 2:
                # Just pick farthest from existing? Or just pick next center?
                # For simplicity, if <2, pick the one that maximizes distance to existing
                if len(placed) == 1:
                    dist = np.linalg.norm(p - placed[0])
                else:
                    dist = 1.0
                score = dist
            else:
                min_tri_area = float('inf')
                for i_p in range(len(placed)):
                    for j_p in range(i_p + 1, len(placed)):
                        p1 = placed[i_p]
                        p2 = placed[j_p]
                        # Triangle area with p, p1, p2
                        area = 0.5 * abs((p1[0] - p[0]) * (p2[1] - p[1]) - (p1[1] - p[1]) * (p2[0] - p[0]))
                        if area < min_tri_area:
                            min_tri_area = area
                score = min_tri_area
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx != -1:
            placed.append(remaining[best_idx])
            remaining = np.delete(remaining, best_idx, axis=0)
            
    return np.array(placed)

def construct():
    start_time = time.time()
    timeout = 110.0
    
    # 1. Greedy Grid Placement
    # Use a moderately fine grid. 20x20 = 400 candidates. 
    # Greedy step: 14 points * 400 candidates * (13*12/2 pairs) ~ 14 * 400 * 78 ~ 400k ops. Fast.
    grid_points = greedy_grid_placement(20, N)
    
    current_points = grid_points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # 2. Simulated Annealing Refinement
    T = 0.05
    T_min = 1e-6
    alpha = 0.9995
    step_size = 0.05
    
    # Random restarts if time permits
    restarts = 0
    max_restarts = 5
    
    while time.time() - start_time < timeout:
        if T < T_min and restarts < max_restarts:
            # Restart with a slightly perturbed version of the best found so far
            # or a new greedy run with a different grid offset
            restarts += 1
            # Perturb best points
            current_points = best_points + np.random.normal(0, 0.1, (N, 2))
            current_score = compute_score(current_points)
            T = 0.02
            step_size = 0.02
            continue
            
        if T < T_min:
            break
            
        new_points = current_points.copy()
        idx = np.random.randint(0, N)
        new_points[idx] += np.random.normal(0, step_size, 2)
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            # Use a scaled delta to avoid underflow/overflow in exp
            delta = new_score - current_score
            if delta > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta / T)
            
            if np.random.rand() < accept_prob:
                current_points = new_points
                current_score = new_score
        
        T *= alpha
        
    return best_points
