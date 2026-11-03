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
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _get_delaunay_neighbors(centres):
    """
    Compute Delaunay triangulation and return a list of neighbor pairs.
    """
    try:
        tri = Delaunay(centres)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(edge)
        return list(edges)
    except Exception:
        # Fallback if Delaunay fails (e.g., collinear points)
        return []


def _local_optimization_delaunay(centres, max_iterations=2000, initial_step=0.01):
    """
    Local optimization using Delaunay neighbor repulsion.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    step_size = initial_step
    
    for iteration in range(max_iterations):
        # Compute current radius
        current_radius = _compute_radius(best_centres)
        if current_radius > best_radius:
            best_radius = current_radius
        
        # Get Delaunay neighbors
        neighbors = _get_delaunay_neighbors(best_centres)
        
        if not neighbors:
            break
            
        # Compute forces
        forces = np.zeros_like(best_centres)
        
        # Repulsion from neighbors
        # We want to increase the minimum distance. 
        # We apply force proportional to (1 - dist / target_dist) for dist < target_dist
        # Target distance is roughly 2 * current_radius * 1.1 to push it out
        target_dist = 2.0 * best_radius * 1.05
        
        for i, j in neighbors:
            dist = np.linalg.norm(best_centres[i] - best_centres[j])
            if dist < 1e-6:
                # Randomly separate coincident points
                diff = np.random.randn(2)
                forces[i] += diff
                forces[j] -= diff
            elif dist < target_dist:
                # Push apart
                direction = (best_centres[i] - best_centres[j]) / dist
                # Strength increases as distance decreases
                strength = (target_dist - dist) / target_dist
                forces[i] += direction * strength
                forces[j] -= direction * strength
        
        # Repulsion from borders
        # If a point is too close to the border, push it in
        border_margin = best_radius * 0.1
        for i in range(len(best_centres)):
            # x-coordinate
            if best_centres[i, 0] < -0.5 + best_radius + border_margin:
                forces[i, 0] += ( -0.5 + best_radius + border_margin - best_centres[i, 0] )
            elif best_centres[i, 0] > 0.5 - best_radius - border_margin:
                forces[i, 0] -= ( best_centres[i, 0] - (0.5 - best_radius - border_margin) )
            
            # y-coordinate
            if best_centres[i, 1] < -0.5 + best_radius + border_margin:
                forces[i, 1] += ( -0.5 + best_radius + border_margin - best_centres[i, 1] )
            elif best_centres[i, 1] > 0.5 - best_radius - border_margin:
                forces[i, 1] -= ( best_centres[i, 1] - (0.5 - best_radius - border_margin) )
        
        # Normalize forces to prevent large jumps
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            forces = forces / max_force
        
        # Apply step
        new_centres = best_centres + forces * step_size
        
        # Project back to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centres = new_centres
            # Decay step size slowly
            step_size *= 0.999
        else:
            # Reduce step size if no improvement
            step_size *= 0.9
            if step_size < 1e-6:
                break
        
        # Optional: Random perturbation to escape local minima
        if iteration % 100 == 50:
            idx = np.random.randint(len(best_centres))
            best_centres[idx] += np.random.randn(2) * step_size * 0.5
            best_centres[idx] = np.clip(best_centres[idx], -0.5, 0.5)
            best_radius = _compute_radius(best_centres)

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
    for restart in range(5):
        # Add random perturbation for restarts > 0
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.01
            centres = np.clip(centres, -0.5, 0.5)
        else:
            # Initial slight perturbation to break symmetry
            centres = centres + np.random.randn(N, 2) * 0.001
            centres = np.clip(centres, -0.5, 0.5)
        
        # Local optimization
        optimised, radius = _local_optimization_delaunay(centres, max_iterations=2000, initial_step=0.01)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
