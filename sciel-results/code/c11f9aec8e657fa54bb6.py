import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute triples
    triples = list(itertools.combinations(range(N), 3))
    triples_arr = np.array(triples)
    p1_idx = triples_arr[:, 0]
    p2_idx = triples_arr[:, 1]
    p3_idx = triples_arr[:, 2]

    def get_score(x):
        points = x.reshape(N, 2)
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area < 1e-9:
            return 0.0
        
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        p3 = points[p3_idx]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    def negative_score_flat(x):
        return -get_score(x)

    # Initial guesses
    best_x = None
    best_score = -np.inf

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Try optimizing x0 with Nelder-Mead (robust to non-smoothness)
    if time.time() < deadline:
        try:
            res = minimize(negative_score_flat, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
        except Exception:
            pass

    # 2. Random Convex Polygons
    rng = np.random.default_rng(42)
    
    # Generate random convex polygons by sorting random points by angle
    # and ensuring they are in convex position (which they are if sorted by angle from centroid)
    # Actually, just picking random points and taking their convex hull vertices?
    # No, we need exactly 14 points.
    # Strategy: Generate 14 points on a circle with random radii? 
    # Or generate 14 points, compute hull, if hull has < 14 points, retry?
    # Better: Generate 14 angles and 14 radii.
    
    for i in range(10):
        if time.time() > deadline - 10: break
        
        # Random angles
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Random radii (log-normal for variety)
        radii = np.exp(rng.normal(0, 0.5, N))
        
        x_start = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Optimize
        try:
            res = minimize(negative_score_flat, x_start, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
        except Exception:
            pass

    # 3. Fine-tuning with L-BFGS-B from the best solution
    if best_x is not None and time.time() < deadline - 5:
        try:
            res = minimize(negative_score_flat, best_x, method='L-BFGS-B',
                           options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
            score = -res.fun
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
        except Exception:
            pass

    if best_x is None:
        best_x = x0

    return best_x
