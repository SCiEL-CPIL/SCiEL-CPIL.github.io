"""New approach: Global optimization using Differential Evolution."""

import numpy as np
from scipy.optimize import differential_evolution

N = 50

def _objective(x):
    """
    Objective function for differential_evolution.
    Minimizes -r, where r is the maximum possible circle radius.
    """
    pts = x.reshape(N, 2)
    
    # 1. Check if points are inside the unit circle (radius 1)
    # The constraint is that circles of radius r must fit.
    # A point at distance d from origin allows a circle of radius 1-d.
    dists_to_origin = np.sqrt(np.sum(pts**2, axis=1))
    r_boundary = 1.0 - np.max(dists_to_origin)
    
    if r_boundary <= 0:
        return 1e6 # Invalid, points outside or on boundary with no room
    
    # 2. Calculate pairwise distances
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distances (diagonal)
    np.fill_diagonal(dist_mat, np.inf)
    
    min_pair_dist = np.min(dist_mat)
    r_pair = min_pair_dist / 2.0
    
    # The actual radius is the minimum of these two constraints
    r = min(r_boundary, r_pair)
    
    # We want to maximize r, so we minimize -r
    return -r

def construct():
    """
    Return the centers of 50 equal circles inside the unit disk.
    Uses Differential Evolution to find a high-quality packing.
    """
    # Bounds for each coordinate: x and y in [-1, 1]
    # We can use slightly tighter bounds to help convergence, e.g., [-0.9, 0.9]
    # Since r ~ 0.125, max dist from origin is ~0.875.
    bounds = [(-0.9, 0.9) for _ in range(2 * N)]
    
    # Run differential evolution
    # maxiter=1000, popsize=20 might be too slow or too fast depending on machine.
    # Let's try popsize=15, maxiter=500. 
    # DE is stochastic, so we might need to run it a few times or with a good seed.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=False # Polish with L-BFGS-B can be slow or stuck, DE usually finds good enough
        )
        best_x = res.x
    except Exception as e:
        # Fallback to a simple spiral if optimization fails
        best_x = np.zeros(2 * N)
        for i in range(N):
            angle = i * 2.399963229728653 # Golden angle
            radius = 0.8 * np.sqrt((i + 0.5) / N)
            best_x[2*i] = radius * np.cos(angle)
            best_x[2*i+1] = radius * np.sin(angle)
            
    return best_x.reshape(N, 2)
