"""
Strategy: Simulated Annealing on the exact radius function.
Unlike the parent's gradient-based soft-min approach, this method treats the
packing problem as a combinatorial optimization task. It uses a Metropolis-style
acceptance criterion to accept moves that decrease the radius if the "temperature"
is high enough, allowing it to escape local minima. The move strategy combines
small Gaussian perturbations with larger "shakes" for specific particles to
break symmetry and explore the configuration space more broadly.
"""

import numpy as np
import time
import math

N = 200
_DEADLINE = 145.0  # Safety margin below 150s kill
_START_TIME = time.time()

def _check_time():
    return time.time() - _START_TIME < _DEADLINE

def _compute_radius(centers):
    """Return the largest common radius for given centers in [-0.5, 0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    # Distance to walls: 0.5 - |x| and 0.5 - |y|
    # Since centers are in [-0.5, 0.5], 0.5 - |x| is distance to nearest vertical wall.
    # But wait, the square is [-0.5, 0.5].
    # Distance to left wall (x=-0.5) is x - (-0.5) = x + 0.5.
    # Distance to right wall (x=0.5) is 0.5 - x.
    # So min dist to x-walls is min(x+0.5, 0.5-x) = 0.5 - |x|.
    # Correct.
    
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
        
    # Pairwise distances
    # Vectorized calculation
    # diff[i,j] = c_i - c_j
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    return min(boundary, min_pair / 2.0)

def _initial_hex_grid(n_points):
    """Generate a hexagonal grid of points."""
    # Approximate hex grid
    # Area per point ~ s^2 * sqrt(3)/2
    # Total area = 1.
    # s = sqrt(2 / (N * sqrt(3)))
    
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    row_height = s * np.sqrt(3) / 2
    n_rows = int(1.0 / row_height) + 4
    n_cols = int(1.0 / s) + 4
    
    # Center the grid
    # We generate a large grid and then select the best N points
    # Or just generate enough and trim.
    
    # Let's generate points in a range slightly larger than the square
    # and then filter.
    
    start_x = -0.5 - s
    start_y = -0.5 - row_height
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * s
            if i % 2 == 1:
                x += s / 2
            y = start_y + i * row_height
            
            # Keep points strictly inside to avoid boundary issues initially
            # We will optimize boundaries later
            if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) == 0:
        # Fallback random
        return np.random.uniform(-0.4, 0.4, size=(n_points, 2))

    # If we have more than N, keep the ones closest to the center?
    # Or just take the first N?
    # Taking closest to center usually yields a denser core.
    if len(points) > n_points:
        center_dist = np.sum(points ** 2, axis=1)
        idx = np.argsort(center_dist)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        # Pad with random points
        extra = np.random.uniform(-0.4, 0.4, size=(n_points - len(points), 2))
        points = np.vstack([points, extra])
        
    return points[:n_points]

def construct():
    start_time = time.time()
    
    # 1. Initialize
    centers = _initial_hex_grid(N)
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.005, size=centers.shape)
    centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
    centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
    
    r_current = _compute_radius(centers)
    r_best = r_current
    best_centers = centers.copy()
    
    # 2. Simulated Annealing
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    # Number of steps
    # We don't know how many steps we can do, so we run until time limit.
    # Estimate: ~0.1ms per step? Maybe more due to O(N^2) radius calc.
    # Let's just loop.
    
    T = T_start
    # Cooling factor per step
    # We want to cool down over a certain number of steps.
    # If we do ~10000 steps, alpha = (T_end/T_start)^(1/10000)
    # But we don't know step count.
    # Let's use a fixed alpha and see.
    alpha = 0.999
    
    # Move size
    step_size = 0.01
    
    # Track stagnation for restarts/shakes
    stagnant_count = 0
    
    while _check_time():
        # Select a random particle to move
        i = np.random.randint(N)
        
        # Generate a move
        # Mix of small moves and occasional larger "jumps"
        if np.random.random() < 0.1:
            # Large shake: move a random subset of particles
            # This helps escape deep local minima
            subset_size = np.random.randint(5, 20)
            idxs = np.random.choice(N, subset_size, replace=False)
            moves = np.random.normal(0, step_size * 2, size=(subset_size, 2))
            new_centers = centers.copy()
            new_centers[idxs] += moves
        else:
            # Single particle move
            move = np.random.normal(0, step_size, size=2)
            new_centers = centers.copy()
            new_centers[i] += move
        
        # Clip to square
        new_centers[:, 0] = np.clip(new_centers[:, 0], -0.499, 0.499)
        new_centers[:, 1] = np.clip(new_centers[:, 1], -0.499, 0.499)
        
        r_new = _compute_radius(new_centers)
        
        # Metropolis criterion
        if r_new > r_current:
            # Always accept improvements
            centers = new_centers
            r_current = r_new
            stagnant_count = 0
            if r_new > r_best:
                r_best = r_new
                best_centers = centers.copy()
        else:
            # Accept worse moves with probability exp((r_new - r_current) / T)
            # Note: r_new - r_current is negative.
            # If T is small, this probability is small.
            delta_r = r_new - r_current
            if delta_r < 0:
                # Avoid underflow
                if T < 1e-10:
                    prob = 0.0
                else:
                    prob = np.exp(delta_r / T)
            else:
                prob = 1.0 # Should not happen if r_new <= r_current, but safe
                
            if np.random.random() < prob:
                centers = new_centers
                r_current = r_new
                stagnant_count += 1
            else:
                stagnant_count += 1
                
        # Cool down
        T *= alpha
        
        # If T is too low, we might get stuck.
        # If stagnant_count is high, we can try to "reheat" or shake.
        if stagnant_count > 1000:
            # Reheat slightly and shake
            T = max(T, 0.01)
            # Apply a global shake to current best? No, to current state.
            # Or just reset current to best and restart with higher T?
            # Let's try a "kick": move all points slightly
            centers += np.random.normal(0, 0.005, size=centers.shape)
            centers[:, 0] = np.clip(centers[:, 0], -0.499, 0.499)
            centers[:, 1] = np.clip(centers[:, 1], -0.499, 0.499)
            r_current = _compute_radius(centers)
            stagnant_count = 0
            
        # Periodic check to see if we should restart from best
        # If we haven't improved best in a long time, maybe restart from best with new T?
        # For now, the shake mechanism should handle it.

    # 3. Final Polish
    # Use a few iterations of steepest descent on the exact gradient?
    # Or just return the best found.
    # The SA might end in a local minimum.
    # Let's try a simple local improvement:
    # For each particle, try moving it in 4 directions + 4 diagonals by small amounts.
    # This is expensive.
    
    # Instead, let's do a few steps of gradient ascent on the soft-min again?
    # No, keep it simple. The SA should have converged.
    
    # Ensure best_centers is valid
    r_final = _compute_radius(best_centers)
    if r_final <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# Dummy main to prevent execution if imported, but construct is the entry point.
if __name__ == "__main__":
    res = construct()
    print(_compute_radius(res))
