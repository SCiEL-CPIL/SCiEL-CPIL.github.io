"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _initial_guess(N):
    """Generate a reasonable initial guess using a hexagonal-like packing pattern."""
    # Start with a grid, then perturb slightly to break symmetry
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].astype(float)
    # Add small random perturbations
    rng = np.random.RandomState(42)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def _objective(points):
    """
    Objective function to minimize. We want to maximize the minimum distance,
    so we minimize the negative of the minimum distance, or equivalently,
    minimize the square of (1 - min_dist) or something similar.
    
    Actually, let's define the objective as:
    For each point, compute the distance to all other points and to the boundaries.
    The "radius" available at point i is half the min of:
      - distance to every other point j
      - distance to each boundary (left, right, bottom, top)
    
    We want to maximize the minimum over all i of these radii.
    
    So we minimize -min_i r_i where r_i = 0.5 * min(min_j dist(i,j), dist_to_boundaries(i))
    
    To make it differentiable-ish, we can use a soft minimum or just work with the raw value.
    Since scipy minimize doesn't handle non-smooth well, let's try a different approach:
    minimize the sum of squared violations of a target radius, or use a penalty method.
    
    Better approach: Use a barrier/penalty method.
    Define objective as:
      sum over all pairs (i,j) of max(0, target_r*2 - dist(i,j))^2
      + sum over all points i and boundaries of max(0, target_r - dist_to_boundary(i))^2
    Then we can do a binary search on target_r, but that's complex.
    
    Alternative: Just minimize -min_radius directly. It's non-smooth but scipy's Nelder-Mead
    or Powell can handle it. Let's try that.
    """
    points = points.reshape(-1, 2)
    n = len(points)
    
    # Compute all pairwise distances
    if n < 2:
        min_pair_dist = 1.0
    else:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
    
    # Distance to boundaries
    # Square is [-0.5, 0.5]^2
    dist_left = points[:, 0] - (-0.5)
    dist_right = 0.5 - points[:, 0]
    dist_bottom = points[:, 1] - (-0.5)
    dist_top = 0.5 - points[:, 1]
    
    dist_to_boundary = np.min(np.stack([dist_left, dist_right, dist_bottom, dist_top], axis=1), axis=1)
    min_boundary_dist = np.min(dist_to_boundary)
    
    # The radius is half the minimum of the two
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    # We want to maximize r, so minimize -r
    return -r

def construct():
    # Initial guess
    points0 = _initial_guess(N)
    
    # Use Nelder-Mead for non-smooth objective
    result = minimize(
        _objective,
        points0.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    points = result.x.reshape(-1, 2)
    
    # Ensure points are within bounds (clip to be safe, but they should already be)
    points = np.clip(points, -0.5, 0.5)
    
    return points
# EVOLVE-BLOCK-END
