"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Initial Guess: 7x7 grid (49 points) + 1 point
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid.copy()
    # Add 50th point at a random location to break symmetry
    points = np.vstack([points, [[0.1, 0.1]]])

    best_points = points.copy()
    best_r = _calculate_radius(points)

    # Gradient Ascent on Min-Radius
    step_size = 0.05
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Calculate min radius and gradients
        r, grad = _get_gradient(points)
        
        # Step
        new_points = points + step_size * grad
        
        # Clip to box
        new_points = np.clip(new_points, -0.5, 0.5)
        
        new_r = _calculate_radius(new_points)
        
        if new_r > best_r:
            best_r = new_r
            best_points = new_points.copy()
            points = new_points
            # Reset step size if improved
            step_size = 0.05
        else:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
            points = best_points.copy()

    return best_points

def _calculate_radius(points):
    n = len(points)
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    d_left = points[:, 0] + 0.5
    d_right = 0.5 - points[:, 0]
    d_bottom = points[:, 1] + 0.5
    d_top = 0.5 - points[:, 1]
    
    min_wall_dist = np.min(np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top))
    
    min_dist = min(min_pair_dist, min_wall_dist)
    return min_dist / 2

def _get_gradient(points):
    n = len(points)
    # Pairwise distances
    dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_left = points[:, 0] + 0.5
    d_right = 0.5 - points[:, 0]
    d_bottom = points[:, 1] + 0.5
    d_top = 0.5 - points[:, 1]
    
    min_pair_dist = np.min(dists)
    min_wall_dist = min(np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top))
    
    min_dist = min(min_pair_dist, min_wall_dist)
    
    grad = np.zeros_like(points)
    
    # Identify pairs at min distance
    # Tolerance to include near-min pairs
    tol = 1e-9
    min_pairs = np.argwhere(dists <= min_pair_dist + tol)
    for i, j in min_pairs:
        if i == j: continue
        d = points[i] - points[j]
        dist = np.linalg.norm(d)
        if dist > 0:
            grad[i] += d / dist
            grad[j] -= d / dist
    
    # Identify walls at min distance
    if min_wall_dist <= min_pair_dist + tol:
        # Left wall
        idx = np.argwhere(d_left <= min_wall_dist + tol)
        for i in idx:
            grad[i, 0] += 1.0
        # Right wall
        idx = np.argwhere(d_right <= min_wall_dist + tol)
        for i in idx:
            grad[i, 0] -= 1.0
        # Bottom wall
        idx = np.argwhere(d_bottom <= min_wall_dist + tol)
        for i in idx:
            grad[i, 1] += 1.0
        # Top wall
        idx = np.argwhere(d_top <= min_wall_dist + tol)
        for i in idx:
            grad[i, 1] -= 1.0
    
    # Normalize gradient
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    norms[norms == 0] = 1
    grad = grad / norms
    
    return min_dist / 2, grad
# EVOLVE-BLOCK-END
