import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _min_triangle_area(points):
    """
    Calculate the minimum area of all triangles formed by triples of points.
    points: numpy array of shape (N, 2)
    """
    n = len(points)
    min_area = np.inf
    # Vectorized calculation for speed
    # For each pair (i, j), compute cross product with all k > j
    # Or simpler: iterate over combinations, but N=13 is small (286 triangles)
    # 286 is very small, so a loop is fine and likely fast enough in Python/NumPy hybrid
    for i in range(n):
        for j in range(i + 1, n):
            v1 = points[j] - points[i]
            # Vectorize over k
            v2s = points - points[i]
            # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
            crosses = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(crosses)
            # We only care about k > j to avoid duplicates, but min is same
            # However, we must exclude i and j themselves (area 0)
            # Since we want min area of distinct triples, we should look at k != i, k != j
            # The loop structure i < j < k is cleaner for correctness
            # Let's stick to the triple loop for N=13, it's only 286 iterations
            pass
    
    # Actually, the triple loop is clearer and 286 is negligible.
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, area is 0
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes the negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = _score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Initial seeds
    seeds = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    seeds.append(pts)
    
    # 2. Center + 12 on circle
    angles_12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    pts_center = np.array([[0.0, 0.0]])
    seeds.append(np.vstack([pts_center, pts_12]))
    
    # 3. Two rings: 1 center, 6 inner, 6 outer
    angles_6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_inner = 0.6 * np.column_stack([np.cos(angles_6), np.sin(angles_6)])
    angles_6_outer = angles_6 + np.pi/6 # Offset
    pts_outer = 1.0 * np.column_stack([np.cos(angles_6_outer), np.sin(angles_6_outer)])
    pts_center = np.array([[0.0, 0.0]])
    seeds.append(np.vstack([pts_center, pts_inner, pts_outer]))
    
    # 4. Three rings: 1 center, 4, 4, 4
    angles_4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pts_r1 = 0.4 * np.column_stack([np.cos(angles_4), np.sin(angles_4)])
    angles_4_2 = angles_4 + np.pi/4
    pts_r2 = 0.8 * np.column_stack([np.cos(angles_4_2), np.sin(angles_4_2)])
    angles_4_3 = angles_4
    pts_r3 = 1.2 * np.column_stack([np.cos(angles_4_3), np.sin(angles_4_3)])
    pts_center = np.array([[0.0, 0.0]])
    seeds.append(np.vstack([pts_center, pts_r1, pts_r2, pts_r3]))
    
    # 5. Random points in unit disk (few restarts)
    rng = np.random.RandomState(42)
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        seeds.append(pts)

    best_points = None
    best_score = -np.inf
    
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
            
        try:
            # Flatten for optimizer
            x0 = seed.flatten()
            
            # Use Nelder-Mead for derivative-free optimization
            # maxiter needs to be reasonable. N=13 -> 26 vars.
            result = minimize(_objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if result.success or result.fun < best_score:
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = result.x.reshape(-1, 2)
                    
        except Exception as e:
            pass
            
    # If no optimization succeeded, fall back to best seed
    if best_points is None:
        best_points = seeds[0]
        
    # Final check and normalization
    # The score is invariant to scaling, but let's ensure the hull area is exactly 1 for consistency
    # Although the evaluator normalizes, it's good practice.
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
        
    return best_points
