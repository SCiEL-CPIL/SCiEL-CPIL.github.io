"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        # Edge case: single circle
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _optimize_gradient(centres, max_iterations=1000, step_size=0.005, tol=0.01):
    """
    Optimizes the centres using a gradient ascent approach on the minimum distance.
    It identifies 'active' constraints (pairs and walls close to the current minimum)
    and applies repulsive forces to increase the minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Adaptive step size
    current_step = step_size
    
    for iteration in range(max_iterations):
        centres = best_centres.copy()
        current_radius = _compute_radius(centres)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
            
        # Determine the target radius for force calculation
        # We want to push points apart if they are closer than best_radius * (1 + tol)
        target_r = best_radius * (1.0 + tol)
        
        forces = np.zeros_like(centres)
        
        # Pairwise forces
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        # Find pairs closer than target_r * 2 (since target_r is radius, distance is 2*r)
        # Actually, let's work with distances directly to avoid confusion.
        # min_pair_dist is the distance. best_radius is dist/2.
        # So we check if dist < 2 * target_r.
        # Wait, _compute_radius returns min(dist/2, wall_dist).
        # So if best_radius is limited by a pair, dist = 2 * best_radius.
        # If limited by wall, wall_dist = best_radius.
        
        # Let's just use the current best_radius to define the "active" set.
        # Any pair with dist < 2 * best_radius * (1+tol) is active.
        # Any point with wall_dist < best_radius * (1+tol) is active.
        
        active_pairs = np.argwhere(dists < 2.0 * target_r)
        
        for i, j in active_pairs:
            if i == j:
                continue
            d = dists[i, j]
            if d < 1e-10:
                # Coincident points, push apart randomly
                dir_vec = np.random.randn(2)
                dir_vec /= np.linalg.norm(dir_vec)
                forces[i] += dir_vec
                forces[j] -= dir_vec
            else:
                dir_vec = (centres[i] - centres[j]) / d
                # Strength proportional to how much they violate the target
                strength = (2.0 * target_r - d)
                forces[i] += strength * dir_vec
                forces[j] -= strength * dir_vec
        
        # Wall forces
        wall_dists = 0.5 - np.abs(centres)
        min_wall_dist_per_point = np.min(wall_dists, axis=1)
        
        active_points = np.argwhere(min_wall_dist_per_point < target_r)
        
        for i, _ in active_points:
            # Determine which wall is closest
            # x is close to 0.5 or -0.5, y is close to 0.5 or -0.5
            # wall_dists[i] is [dist_to_right, dist_to_top, dist_to_left, dist_to_bottom]?
            # No, np.abs(centres) gives [abs(x), abs(y)].
            # 0.5 - abs(x) is min dist to x-walls.
            # 0.5 - abs(y) is min dist to y-walls.
            
            dx = 0.5 - np.abs(centres[i, 0])
            dy = 0.5 - np.abs(centres[i, 1])
            
            force_vec = np.zeros(2)
            
            if dx < target_r:
                # Push away from x-wall
                # If x > 0, push left (-1). If x < 0, push right (1).
                sign_x = -np.sign(centres[i, 0])
                if centres[i, 0] == 0:
                    sign_x = np.random.choice([-1, 1])
                force_vec[0] += (target_r - dx) * sign_x
            
            if dy < target_r:
                # Push away from y-wall
                sign_y = -np.sign(centres[i, 1])
                if centres[i, 1] == 0:
                    sign_y = np.random.choice([-1, 1])
                force_vec[1] += (target_r - dy) * sign_y
            
            forces[i] += force_vec
        
        # Normalize forces to prevent huge jumps
        norm_forces = np.linalg.norm(forces, axis=1)
        max_norm = np.max(norm_forces)
        if max_norm > 1e-10:
            # Scale forces so the max step is current_step
            # This ensures stability
            forces = forces / norm_forces[:, np.newaxis] * current_step
        else:
            break
        
        # Update positions
        new_centres = centres + forces
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Check if we improved
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + 1e-10:
            best_radius = new_radius
            best_centres = new_centres
            # Increase step size slightly if we are making good progress?
            # Or keep it constant.
        else:
            # No improvement, reduce step size
            current_step *= 0.95
            if current_step < 1e-6:
                break
                
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run multiple iterations of local optimization with different random restarts
    np.random.seed(42) # For reproducibility in this block, though construct might be called multiple times
    
    for restart in range(5):
        # Add small random perturbation to escape local minima
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.01
            centres = np.clip(centres, -0.5, 0.5)
        else:
            centres = best_centres.copy()
            
        # Local optimization
        optimised, radius = _optimize_gradient(centres, max_iterations=2000, step_size=0.002, tol=0.05)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
