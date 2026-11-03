"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _compute_gradients(centres, active_pairs, active_borders):
    """
    Compute gradient of the minimum distance objective.
    active_pairs: list of (i, j) pairs that are at or near minimum distance
    active_borders: list of (i, k) where k is 0 or 1 (x or y border)
    Returns gradient array of shape (N, 2)
    """
    N = len(centres)
    grad = np.zeros((N, 2))
    
    # Pair contributions: push apart
    for i, j in active_pairs:
        if i == j:
            continue
        diff = centres[i] - centres[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            diff = np.random.randn(2)
            norm = np.linalg.norm(diff)
        diff = diff / norm
        grad[i] += 0.5 * diff
        grad[j] -= 0.5 * diff
    
    # Border contributions: push inward
    for i, k in active_borders:
        if k == 0:  # x border
            if centres[i][0] > 0:
                grad[i][0] -= 0.5
            else:
                grad[i][0] += 0.5
        else:  # y border
            if centres[i][1] > 0:
                grad[i][1] -= 0.5
            else:
                grad[i][1] += 0.5
    
    return grad


def _local_optimization(centres, max_iterations=2000, initial_step=0.01, tolerance=1e-12):
    """
    Improved local optimization using gradient descent on the min-distance objective.
    Uses a simulated annealing schedule to escape local minima.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    current_radius = best_radius
    
    step = initial_step
    patience = 0
    max_patience = 200
    
    for iteration in range(max_iterations):
        # Compute current state
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_border_dist = np.min(0.5 - np.abs(centres))
        current_radius = min(min_pair_dist / 2.0, min_border_dist)
        
        # Find all pairs within 2% of minimum (active constraints)
        threshold_pair = min_pair_dist * 1.02
        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= threshold_pair:
                    active_pairs.append((i, j))
        
        # Find all border constraints within 2% of minimum
        border_dists = 0.5 - np.abs(centres)
        threshold_border = min_border_dist * 1.02
        active_borders = []
        for i in range(N):
            if border_dists[i, 0] <= threshold_border:
                active_borders.append((i, 0))
            if border_dists[i, 1] <= threshold_border:
                active_borders.append((i, 1))
        
        # Compute gradient
        grad = _compute_gradients(centres, active_pairs, active_borders)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Take step
        new_centres = centres + grad * step
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        
        # Accept if improved, or with some probability if not (simulated annealing)
        if new_radius >= current_radius:
            centres = new_centres
            current_radius = new_radius
            patience = 0
        else:
            # Simulated annealing: accept with probability exp(-delta / temperature)
            temperature = step * 0.1
            delta = current_radius - new_radius
            if delta < 1e-15:
                centres = new_centres
                current_radius = new_radius
                patience = 0
            else:
                accept_prob = np.exp(-delta / max(temperature, 1e-15))
                if np.random.rand() < accept_prob:
                    centres = new_centres
                    current_radius = new_radius
                    patience = 0
                else:
                    patience += 1
        
        # Update best
        if current_radius > best_radius + 1e-15:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Adaptive step size
        if patience > 50:
            step *= 0.95
            patience = 0
        
        # Termination criteria
        if step < 1e-10:
            break
        if patience > max_patience:
            break
    
    return best_centres, best_radius


def _polish(centres, max_iterations=5000, initial_step=0.001):
    """
    Final polishing phase with very small steps to fine-tune the packing.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    current_radius = best_radius
    
    step = initial_step
    
    for iteration in range(max_iterations):
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_border_dist = np.min(0.5 - np.abs(centres))
        current_radius = min(min_pair_dist / 2.0, min_border_dist)
        
        # Find all pairs within 1% of minimum
        threshold_pair = min_pair_dist * 1.01
        active_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= threshold_pair:
                    active_pairs.append((i, j))
        
        # Find all border constraints within 1% of minimum
        border_dists = 0.5 - np.abs(centres)
        threshold_border = min_border_dist * 1.01
        active_borders = []
        for i in range(N):
            if border_dists[i, 0] <= threshold_border:
                active_borders.append((i, 0))
            if border_dists[i, 1] <= threshold_border:
                active_borders.append((i, 1))
        
        # Compute gradient
        grad = _compute_gradients(centres, active_pairs, active_borders)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Take small step
        new_centres = centres + grad * step
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        
        if new_radius >= current_radius:
            centres = new_centres
            current_radius = new_radius
            if current_radius > best_radius + 1e-15:
                best_radius = current_radius
                best_centres = centres.copy()
        
        # Very slow step decay
        step *= 0.999
        
        if step < 1e-12:
            break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    np.random.seed(42)
    
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Multiple restarts with increasing refinement
    for restart in range(5):
        if restart == 0:
            start_centres = centres.copy()
        else:
            # Perturb from best so far
            perturb_scale = 0.01 * (1.0 - restart * 0.15)
            start_centres = best_centres + np.random.randn(N, 2) * perturb_scale
            start_centres = np.clip(start_centres, -0.5, 0.5)
        
        # Local optimization
        optimised, radius = _local_optimization(start_centres, max_iterations=1500, initial_step=0.008)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    # Final polishing phase
    polished, final_radius = _polish(best_centres, max_iterations=3000, initial_step=0.0005)
    
    if final_radius > best_radius:
        best_radius = final_radius
        best_centres = polished.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
