import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar

N = 100

def construct():
    def generate_fcc_points(a, n_range):
        points = []
        for i in range(-n_range, n_range + 1):
            for j in range(-n_range, n_range + 1):
                for k in range(-n_range, n_range + 1):
                    if (i + j + k) % 2 == 0:
                        points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
        return np.array(points)
    
    def radius_for_a(a):
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            return 0.0
        
        dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        order = np.argsort(dist_to_boundary)[::-1]
        selected = points[order[:N]]
        
        tree = cKDTree(selected)
        dists, _ = tree.query(selected, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        min_boundary_dist = np.min(dist_to_boundary[order[:N]])
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Optimize a
    # Initial guess around 0.3
    result = minimize_scalar(lambda a: -radius_for_a(a), bounds=(0.25, 0.35), method='bounded')
    best_a = result.x
    best_points = None
    best_score = 0.0
    
    # Re-generate points with best_a to return them
    n_range = int(np.ceil(0.5 / (best_a / 2.0))) + 2
    points = generate_fcc_points(best_a, n_range)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        order = np.argsort(dist_to_boundary)[::-1]
        best_points = points[order[:N]]
        best_score = radius_for_a(best_a)
        
    if best_points is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
