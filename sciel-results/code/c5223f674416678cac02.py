"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Precompute triangle vertices and indices for area calculation
V = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
indices = list(itertools.combinations(range(N), 3))
idx_array = np.array(indices)

def min_area_fast(points):
    """Calculate the minimum area of all triangles formed by the points."""
    p1 = points[idx_array[:, 0]]
    p2 = points[idx_array[:, 1]]
    p3 = points[idx_array[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # y >= 0
    pen1 = np.sum(np.maximum(0, -y)**2)
    # y <= sqrt(3) x
    pen2 = np.sum(np.maximum(0, y - np.sqrt(3)*x)**2)
    # sqrt(3) x + y <= sqrt(3)
    pen3 = np.sum(np.maximum(0, np.sqrt(3)*x + y - np.sqrt(3))**2)
    return pen1 + pen2 + pen3

def objective(flat_points):
    """Objective function to minimize: -min_area + penalty."""
    points = flat_points.reshape(N, 2)
    area = min_area_fast(points)
    pen = penalty(points)
    return -area + 100 * pen

def random_points():
    """Generate N random points inside the triangle."""
    pts = []
    for _ in range(N):
        u = np.random.random()
        v = np.random.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        pts.append(u * V[1] + v * V[2])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial best is a random configuration
    best_points = random_points()
    best_score = objective(best_points.flatten())
    
    np.random.seed(42)
    
    max_restarts = 200
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        x0 = random_points().flatten()
        
        # Run Nelder-Mead local optimization
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_score:
            best_score = res.fun
            best_points = res.x.reshape(N, 2)
            
    return best_points
# EVOLVE-BLOCK-END
