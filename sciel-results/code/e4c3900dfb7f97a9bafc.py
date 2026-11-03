"""Improved program using Voronoi-based sparse optimization for circle packing."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_delaunay_edges(centers):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        simplices = tri.simplices
        edges = set()
        for simplex in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _get_voronoi_inradii(centers):
    """
    Compute the inradius of each Voronoi cell for each circle center.
    The inradius of a Voronoi cell is the distance from the center to the nearest
    edge of the cell, which equals half the distance to the nearest neighbor in
    the Delaunay triangulation, or the distance to the wall if closer.
    """
    # Get Delaunay neighbors
    edges = _get_delaunay_edges(centers)
    
    # Build adjacency
    neighbors = [[] for _ in range(N)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    
    # For each point, compute the minimum distance to its Delaunay neighbors / 2
    # and to the walls
    inradii = np.zeros(N)
    
    for i in range(N):
        # Distance to walls
        wall_dist = 0.5 - np.abs(centers[i])
        min_r = np.min(wall_dist)
        
        # Distance to Delaunay neighbors
        for j in neighbors[i]:
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            if d < min_r:
                min_r = d
        
        inradii[i] = min_r
    
    return inradii

def _optimize_from_start(start_centers, bounds):
    """
    Optimization using Voronoi-aware sparse objective.
    The objective is to maximize the minimum inradius of all Voronoi cells.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # Helper to compute sparse objective based on Voronoi inradii
    def objective(params, k):
        centers = params.reshape(N, 2)
        # Keep centers within bounds to avoid weird gradients
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Compute Voronoi inradii
        inradii = _get_voronoi_inradii(centers)
        
        if len(inradii) == 0:
            return 0.0
            
        m = np.min(inradii)
        # Soft min to make it differentiable
        shifted = inradii - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    # Use L-BFGS-B with the sparse objective
    try:
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            args=(500.0,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
        
    # Second pass with higher k for polishing
    try:
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            args=(2000.0,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        centers = result.x.reshape(N, 2)
        centers = np.clip(centers, -0.4999, 0.4999)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
        
    return best_centers, best_score

def construct():
    bounds = [(-0.4999, 0.4999)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid
    s = 0.14 
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.02, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0: 
            break
            
        centers, score = _optimize_from_start(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
