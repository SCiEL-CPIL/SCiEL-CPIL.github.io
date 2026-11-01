"""Improved packing of 50 equal circles in a unit square via simulated annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    if len(centres) < 2:
        min_pair = np.inf
    else:
        min_pair = pdist(centres, metric='euclidean').min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _construct_grid():
    """Create a grid-based initial configuration."""
    m = 7  # 7x7 = 49, need 50, so use 7x8 or similar
    # Use a 7x7 grid plus one extra, or 8x7
    # Let's use a 7x7 grid (49 points) and add one more
    # Actually, let's try different grids
    
    # 7x7 grid gives 49 points
    ticks_x = np.linspace(-0.5 + 1/14, 0.5 - 1/14, 7)
    ticks_y = np.linspace(-0.5 + 1/14, 0.5 - 1/14, 7)
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # We need 50 points. Add one more point at a reasonable location
    # Or use a different grid. Let's use 8 columns and 7 rows but only take 50
    ticks_x2 = np.linspace(-0.5 + 1/16, 0.5 - 1/16, 8)
    ticks_y2 = np.linspace(-0.5 + 1/14, 0.5 - 1/14, 7)
    grid2 = np.array([(x, y) for y in ticks_y2 for x in ticks_x2])
    
    return grid2[:N]

def _optimize(centres, max_iter=2000, seed=42):
    """Optimize circle packing using gradient-free optimization."""
    np.random.seed(seed)
    random.seed(seed)
    
    best_centres = centres.copy()
    best_score = _radius_score(centres)
    
    T = 0.01  # initial temperature
    cooling = 0.999
    
    for it in range(max_iter):
        # Generate a perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, T, size=2)
        new_centres = centres.copy()
        new_centres[idx] += delta
        
        # Clamp to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_score = _radius_score(new_centres)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score >= best_score:
            best_centres = new_centres.copy()
            best_score = new_score
            centres = new_centres
        else:
            prob = np.exp((new_score - best_score) / T)
            if random.random() < prob:
                centres = new_centres
        
        T *= cooling
    
    return best_centres, best_score

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    # Start with a grid configuration
    centres = _construct_grid()
    
    # Run optimization with multiple random restarts
    best_centres = centres.copy()
    best_score = _radius_score(centres)
    
    for seed in range(10):
        # Perturb initial configuration
        np.random.seed(seed + 100)
        perturbed = centres + np.random.normal(0, 0.02, size=(N, 2))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        opt_centres, opt_score = _optimize(perturbed, max_iter=1500, seed=seed + 200)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centres = opt_centres
    
    return best_centres
# EVOLVE-BLOCK-END
